"""Real same-image spatial questions: paired cached/serial latency and benchmark quality."""

import argparse
import json
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from spatial_jev.batching import make_processor  # noqa: E402
from spatial_jev.generation_eval import parse_generated_answer  # noqa: E402
from spatial_jev.inference import JevSpatial  # noqa: E402
from spatial_jev.model import build_model  # noqa: E402
from spatial_jev.official_benchmarks import parse_points  # noqa: E402
from spatial_jev.robospatial_benchmarks import official_module, parse_yes_no  # noqa: E402
from spatial_jev.runtime import InferenceBuilder  # noqa: E402
from spatial_jev.scene_inference import ScenePrefix, predict_scene  # noqa: E402
from spatial_jev.schema import compile_prompt  # noqa: E402
from benchmark_latency import sha  # noqa: E402
from evaluate_fast_benchmarks import native_official_predict, native_predict  # noqa: E402

METHODS = ['native_serial', 'native_shared', 'three_head_shared', 'unified_serial', 'unified_shared']


def make_scenes(count, seed):
    rng = random.Random(seed)
    scenes = []
    for suite, source in [('robospatial', 'robospatial_v1'), ('numeric', 'fast_v1')]:
        groups = defaultdict(dict)
        path = ROOT / 'data/benchmarks' / source / 'requests.jsonl'
        for row in map(json.loads, path.open()):
            if suite == 'numeric' and row['benchmark'] != 'vst_numeric_dev':
                continue
            image = tuple(m['sha256'] for m in row['input']['media'])
            groups[image][row['input']['question']] = row
        eligible = []
        for key, values in groups.items():
            rows = list(values.values())
            point = [r for r in rows if r['input']['answer_space']['kind'] == 'point']
            choices = [r for r in rows if r['input']['answer_space']['kind'] == 'choice']
            if (suite == 'numeric' and len(rows) >= 2) or (
                    suite == 'robospatial' and len(rows) >= 8 and len(point) >= 2 and len(choices) >= 4):
                eligible.append((key, rows))
        chosen = rng.sample(sorted(eligible), count)
        for index, (key, rows) in enumerate(chosen):
            rng.shuffle(rows)
            if suite == 'robospatial':
                point = [r for r in rows if r['family'] == 'context']
                choices = [r for r in rows if r['family'] != 'context']
                # The first four always include two placement requests and two VQ decisions.
                a, b = choices[:2], point[:2]
                ordered = [a[0], b[0], a[1], b[1]] if index % 2 == 0 else [b[0], a[0], b[1], a[1]]
                selected_ids = {r['id'] for r in ordered}
                ordered += [r for r in rows if r['id'] not in selected_ids][:4]
                sizes = [1, 2, 4, 8]
            else:
                ordered, sizes = rows[:2], [1, 2]
            scenes.append({'scene_id': suite + ':' + key[0][:16], 'suite': suite,
                           'sizes': sizes, 'questions': ordered})
    return scenes


def native_text(row, builder):
    inp = row['input']
    kind = inp['answer_space']['kind']
    if kind == 'scalar':
        text, _ = compile_prompt(inp)
        text += '\nAnswer with only the number and unit.'
    elif kind == 'point':
        text = re.sub(r'\b(?:several|multiple|a few|some)\s+(?:points|spots|locations)\b',
                      'one point', inp['question'], flags=re.I)
        text += '\nPoint to exactly one valid location. Use your usual coordinate output format.'
    else:
        text = row['native_prompt']
    return builder.render(text, len(inp['media']))


def native_shared(model, builder, rows, device):
    images = builder.load_images(rows[0]['input'])
    scene = ScenePrefix(model.model, builder, images, [native_text(r, builder) for r in rows], device)
    hidden = scene.forward(dict(enumerate(scene.suffixes)))
    active = list(range(len(rows)))
    generated = [[] for _ in rows]
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    limit = 128 if rows[0]['benchmark'] == 'vst_numeric_dev' else 256
    for step in range(limit):
        tokens = model.lm_head(hidden).float().argmax(-1).tolist()
        next_ids = {}
        for owner, token in zip(active, tokens):
            generated[owner].append(token)
            if token not in eos:
                next_ids[owner] = torch.tensor([[token]], device=device)
        if not next_ids or step == limit - 1:
            break
        active = list(next_ids)
        hidden = scene.forward(next_ids)
    results = []
    for row, tokens in zip(rows, generated):
        raw = builder.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        out = {'raw_answer': raw, 'output_tokens': len(tokens),
               'generation_truncated': len(tokens) == limit and tokens[-1] not in eos}
        try:
            task = row['input']['answer_space']['kind']
            if task == 'choice':
                out['prediction'] = parse_yes_no(raw)
            elif task == 'scalar':
                out['prediction'], _ = parse_generated_answer(raw, row['input'], [])
            else:
                m = row['input']['media'][0]
                out['points'], _ = parse_points(raw, m['width'], m['height'])
                out['prediction'] = out['points'][0]
            out['parse_error'] = None
        except ValueError as exc:
            out.update(prediction=None, points=[], parse_error=str(exc))
        results.append(out)
    return results


def typed_shared(model, builder, rows, device):
    prompts = [compile_prompt(row['input']) for row in rows]
    images = builder.load_images(rows[0]['input'])
    scene = ScenePrefix(model.backbone, builder, images,
                        [builder.render(p[0], len(images)) for p in prompts], device)
    hidden = scene.forward(dict(enumerate(scene.suffixes)))
    result = [None] * len(rows)
    for task in ['choice', 'scalar', 'point']:
        indices = [i for i, row in enumerate(rows) if row['input']['answer_space']['kind'] == task]
        if not indices:
            continue
        counts = torch.tensor([len(prompts[i][1]) for i in indices], device=device) if task == 'choice' else None
        with torch.autocast('cuda', enabled=False):
            values = model.heads(hidden[indices].float(), task, counts)
        for i, value in zip(indices, values):
            prediction = (prompts[i][1][int(value.argmax())] if task == 'choice' else
                          value.clamp(max=20).expm1().item() if task == 'scalar' else value.cpu().tolist())
            result[i] = {'prediction': prediction}
    return result


def score_all(scenes, records):
    # Ground truth is opened only after every prediction and timing has been written.
    refs = {}
    for suite in ['robospatial_v1', 'fast_v1']:
        refs.update({r['id']: r for r in map(json.loads, (ROOT / 'data/benchmarks' / suite / 'references.jsonl').open())})
    requests = {r['id']: r for scene in scenes for r in scene['questions']}
    summaries = {}
    for suite in ['robospatial', 'numeric']:
        if not any(s['suite'] == suite for s in scenes):
            continue
        summaries[suite] = {}
        sizes = [1, 2, 4, 8] if suite == 'robospatial' else [1, 2]
        for size in sizes:
            summaries[suite][size] = {}
            for method in METHODS:
                selected = [r for r in records if r['suite'] == suite and r['size'] == size and r['method'] == method]
                by_scene = defaultdict(list)
                for r in selected:
                    by_scene[r['scene_id']].append(r)
                times = [statistics.median(r['milliseconds'] for r in group) for group in by_scene.values()]
                scores = defaultdict(list)
                valid, n = 0, 0
                for group in by_scene.values():
                    sample = group[0]
                    for key, pred in zip(sample['question_ids'], sample['outputs']):
                        row, target = requests[key], refs[key]['target']
                        task, value = row['input']['answer_space']['kind'], pred.get('prediction')
                        n += 1
                        valid += value is not None
                        if task == 'choice':
                            scores['choice_accuracy'].append(float(value == target['option_id']))
                        elif task == 'scalar':
                            if value is not None:
                                scores['scalar_mae_m'].append(abs(value - target['value']))
                        else:
                            text = repr([tuple(value)]) if value is not None else '[]'
                            ok, _, _, _ = official_module().evaluate_answer(
                                target['source_answer'], text, mask_path=target['mask_path'],
                                category='context', num_points_to_match=1)
                            scores['point_mask_success'].append(float(ok))
                summaries[suite][size][method] = {
                    'scenes': len(times), 'mean_ms': statistics.mean(times),
                    'p50_ms': float(np.quantile(times, .5)), 'p95_ms': float(np.quantile(times, .95)),
                    'ms_per_question': statistics.mean(times) / size,
                    'decisions_per_second': size * 1000 / statistics.mean(times),
                    'requests': n, 'valid_rate': valid / n,
                    **{key: statistics.mean(values) for key, values in scores.items()}}
    return summaries


def profile_call(backbone, function, device):
    """Separate diagnostic pass; phase hooks are absent from the main latency measurements."""
    spans, vision, starts = [], [], {}

    def begin(module, args, kwargs):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        cache = kwargs.get('past_key_values')
        starts[id(module)] = (event, cache is None or cache.get_seq_length() == 0)

    def finish(module, args, kwargs, result):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        start, initial = starts.pop(id(module))
        (vision if module is backbone.vision_backbone else spans).append((start, event, initial))

    handles = []
    for module in [backbone, backbone.vision_backbone]:
        handles += [module.register_forward_pre_hook(begin, with_kwargs=True),
                    module.register_forward_hook(finish, with_kwargs=True)]
    try:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        function()
        torch.cuda.synchronize(device)
        total = (time.perf_counter() - started) * 1000
    finally:
        for handle in handles:
            handle.remove()
    initial = sum(a.elapsed_time(b) for a, b, first in spans if first)
    later = sum(a.elapsed_time(b) for a, b, first in spans if not first)
    return {'total_ms': total, 'initial_forward_ms': initial, 'later_forward_ms': later,
            'vision_ms_included_in_forwards': sum(a.elapsed_time(b) for a, b, _ in vision),
            'other_ms_including_preprocessing_readout': total - initial - later,
            'backbone_calls': len(spans), 'vision_calls': len(vision)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenes', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--spatial-crops', type=int, choices=[2, 24], default=24)
    parser.add_argument('--suite', choices=['all', 'robospatial', 'numeric'], default='all')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    scenes = make_scenes(args.scenes, args.seed)
    if args.suite != 'all':
        scenes = [s for s in scenes if s['suite'] == args.suite]
    suite_names = sorted({s['suite'] for s in scenes})
    (args.output / 'scenes.json').write_text(json.dumps(scenes, indent=2) + '\n')
    model_path = ROOT / 'releases/publication-v0.1.0/huggingface'
    base = ROOT / 'models/allenai/Molmo2-ER'
    checkpoint = ROOT / 'outputs/pilot_v0/20260923T161433Z/full/checkpoint-001000'
    config = json.loads((checkpoint / 'config.json').read_text())
    shared = JevSpatial.from_pretrained(model_path, device=device, preserve_option_order=True)
    native = transformers.AutoModelForImageTextToText.from_pretrained(
        base, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    typed = build_model(base, config, checkpoint, trainable=False).to(device)
    typed.backbone = typed.backbone.merge_and_unload(safe_merge=True)
    typed.eval()
    processor = make_processor(base, 24)
    configs = {c: {**config, 'max_crops': c, 'max_sequence_length': limit,
                   'native_point_policy': 'single', 'robospatial_vq_prompt': 'source', 'preserve_option_order': True}
               for c, limit in [(2, 4096), (24, 64000)]}
    builders = {c: InferenceBuilder(processor, {**shared.config, **cfg}) for c, cfg in configs.items()}
    runners = {c: JevSpatial(shared.model, shared.processor, {**shared.config, **cfg}, device)
               for c, cfg in configs.items()}
    meta = {'seed': args.seed, 'scenes_per_suite': args.scenes, 'repeats': args.repeats,
            'gpu': torch.cuda.get_device_name(device), 'device': str(device),
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'precision': 'BF16 backbone, FP32 heads', 'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cpu_threads': 4,
            'weights': 'same native, merged three-head and merged unified checkpoints as the single-question comparison',
            'scene_sha256': sha(args.output / 'scenes.json'),
            'script_sha256': sha(__file__), 'scene_inference_sha256': sha(ROOT / 'src/spatial_jev/scene_inference.py'),
            'timing': 'warm, synchronized E2E wall time including IO/preprocessing/readout/decoding; batch-one scene request',
            'native_shared': 'same individual prompts; shared image KV; independent greedy decoding branches processed together',
            'unified_shared': 'shared image KV, isolated packed questions; all active crop/refinement paths batched at each level',
            'serial_baselines': 'sum measured per-question times for each nested request size, in each repetition',
            'profiles': {'robospatial': f'{args.spatial_crops} crops / {configs[args.spatial_crops]["max_sequence_length"]} tokens',
                         'numeric': '2 crops / 4096 tokens'},
            'native_point_policy': 'one normalized point', 'methods': METHODS}
    (args.output / 'config.json').write_text(json.dumps(meta, indent=2) + '\n')

    def setup(scene):
        c = 2 if scene['suite'] == 'numeric' else args.spatial_crops
        processor.image_processor.max_crops = c
        shared.processor.image_processor.max_crops = c
        return configs[c], builders[c], runners[c]

    def call(method, rows, cfg, builder, runner):
        if method == 'native_shared':
            return native_shared(native, builder, rows, device)
        if method == 'three_head_shared':
            return typed_shared(typed, builder, rows, device)
        if method == 'unified_shared':
            return predict_scene(runner, [r['input'] for r in rows], args.seed)
        if method == 'unified_serial':
            return [runner.predict(r['input'], seed=args.seed) for r in rows]
        return [(native_predict(native, processor, r['input'], cfg, device)
                 if r['benchmark'] == 'vst_numeric_dev' else
                 native_official_predict(native, processor, r, cfg, device)) for r in rows]

    records, parity = [], []
    order_rng = random.Random(args.seed + 1)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for suite in suite_names:
            scene = next(s for s in scenes if s['suite'] == suite)
            cfg, builder, runner = setup(scene)
            for method in METHODS:
                call(method, scene['questions'][:2], cfg, builder, runner)
        print('Warmup done', flush=True)
        with (args.output / 'measurements.jsonl').open('w') as stream:
            for index, scene in enumerate(scenes):
                cfg, builder, runner = setup(scene)
                baseline = {}
                # Original outputs and source prompts form an independent semantic check.
                for method in ['native_serial', 'unified_serial']:
                    per_repeat = []
                    for repeat in range(1 if args.verify_only else args.repeats):
                        times, outputs = [], []
                        for row in scene['questions']:
                            torch.cuda.synchronize(device)
                            start = time.perf_counter()
                            out = call(method, [row], cfg, builder, runner)[0]
                            torch.cuda.synchronize(device)
                            times.append((time.perf_counter() - start) * 1000)
                            outputs.append(out)
                        per_repeat.append(outputs)
                        for size in scene['sizes']:
                            record = {'scene_id': scene['scene_id'], 'suite': scene['suite'], 'size': size,
                                      'method': method, 'repeat': repeat, 'milliseconds': sum(times[:size]),
                                      'question_ids': [r['id'] for r in scene['questions'][:size]], 'outputs': outputs[:size]}
                            records.append(record)
                            stream.write(json.dumps(record) + '\n')
                    baseline[method] = per_repeat[0]
                methods = ['native_shared', 'three_head_shared', 'unified_shared']
                for size in scene['sizes']:
                    for repeat in range(1 if args.verify_only else args.repeats):
                        order_rng.shuffle(methods)
                        for method in methods:
                            rows = scene['questions'][:size]
                            torch.cuda.synchronize(device)
                            start = time.perf_counter()
                            outputs = call(method, rows, cfg, builder, runner)
                            torch.cuda.synchronize(device)
                            ms = (time.perf_counter() - start) * 1000
                            record = {'scene_id': scene['scene_id'], 'suite': scene['suite'], 'size': size,
                                      'method': method, 'repeat': repeat, 'milliseconds': ms,
                                      'question_ids': [r['id'] for r in rows], 'outputs': outputs}
                            records.append(record)
                            stream.write(json.dumps(record) + '\n')
                            stream.flush()
                            if repeat == 0 and method != 'three_head_shared':
                                old = baseline['native_serial' if method == 'native_shared' else 'unified_serial']
                                for row, before, after in zip(rows, old, outputs):
                                    parity.append({'id': row['id'], 'size': size, 'method': method,
                                                   'task': row['input']['answer_space']['kind'],
                                                   'same_prediction': before['prediction'] == after['prediction'],
                                                   'before': before['prediction'], 'after': after['prediction']})
                print(json.dumps({'scene': index + 1, 'total': len(scenes), 'suite': scene['suite']}), flush=True)
    (args.output / 'parity.json').write_text(json.dumps(parity, indent=2) + '\n')
    if not args.verify_only:
        phases = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for suite in suite_names:
                for scene in [s for s in scenes if s['suite'] == suite][:3]:
                    cfg, builder, runner = setup(scene)
                    for method in METHODS:
                        backbone = (native.model if method.startswith('native') else typed.backbone
                                    if method.startswith('three_head') else shared.model.backbone)
                        measurement = profile_call(backbone, lambda: call(
                            method, scene['questions'], cfg, builder, runner), device)
                        phases.append({'scene_id': scene['scene_id'], 'suite': suite, 'method': method,
                                       'questions': len(scene['questions']), **measurement})
        (args.output / 'phases.json').write_text(json.dumps(phases, indent=2) + '\n')
    summary = {'config': meta, 'results': score_all(scenes, records),
               'parity': {method: {'n': len(rows), 'exact_rate': statistics.mean(r['same_prediction'] for r in rows)}
                          for method in ['native_shared', 'unified_shared']
                          if (rows := [r for r in parity if r['method'] == method])}}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
