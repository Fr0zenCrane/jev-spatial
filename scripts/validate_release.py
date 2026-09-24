"""Full legacy/clean/merged parity checks with a fixed, target-free request manifest."""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from spatial_jev.inference import JevSpatial  # noqa: E402
from spatial_jev.unified import UnifiedBuilder, build_unified_model, predict_unified  # noqa: E402
from spatial_jev.batching import make_processor  # noqa: E402


def summarize(rows):
    result = {'n': len(rows), 'clean_exact': all(r['clean_exact'] for r in rows), 'by_task': {}}
    for task in ['choice', 'scalar', 'point']:
        selected = [r for r in rows if r['task'] == task]
        if not selected:
            continue
        errors = [r['output_distance'] for r in selected]
        deltas = [r['max_same_prefix_logit_delta'] for r in selected]
        group = {'n': len(selected), 'exact_predictions': sum(r['same_prediction'] for r in selected),
                 'exact_prediction_rate': np.mean([r['same_prediction'] for r in selected]).item(),
                 'first_decision_agreement': np.mean([r['before']['path'][0] == r['after']['path'][0] for r in selected]).item(),
                 'output_distance_mean': float(np.mean(errors)), 'output_distance_p95': float(np.quantile(errors, .95)),
                 'output_distance_max': max(errors), 'max_same_prefix_logit_delta': max(deltas)}
        result['by_task'][task] = group
    # Criteria are fixed before the full run. Exact clean-code equality is mandatory.
    choice = result['by_task'].get('choice', {})
    scalar = result['by_task'].get('scalar', {})
    point = result['by_task'].get('point', {})
    result['checks'] = {
        'clean_code_exact': result['clean_exact'],
        'choice_agreement_ge_99pct': choice.get('exact_prediction_rate', 1) >= .99,
        'scalar_first_decision_ge_99pct': scalar.get('first_decision_agreement', 1) >= .99,
        'scalar_exact_value_ge_95pct': scalar.get('exact_prediction_rate', 1) >= .95,
        'point_first_decision_ge_98pct': point.get('first_decision_agreement', 1) >= .98,
        'point_exact_coordinate_ge_95pct': point.get('exact_prediction_rate', 1) >= .95,
        'point_p95_distance_le_one_cell_diagonal': point.get('output_distance_p95', 0) <= math.sqrt(2)/27,
    }
    result['passed'] = all(result['checks'].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit-per-task', type=int, default=0)
    args = parser.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    torch.set_num_threads(4)
    config = json.loads((args.checkpoint / 'config.json').read_text())
    requests = []
    for line in (ROOT / 'data/processed/pilot_v0/20260923T160355Z/dev.jsonl').open():
        row = json.loads(line)
        requests.append({'id': 'pilot:' + row['sample_id'], 'input': row['input'], 'profile': 'original_default',
                         'seed': config['seed'] ^ int(row['sample_id'][:8], 16)})
    for name in ['official_images_v1', 'robospatial_v1']:
        for line in (ROOT / 'data/benchmarks' / name / 'requests.jsonl').open():
            row = json.loads(line)
            requests.append({'id': name + ':' + row['id'], 'input': row['input'], 'profile': 'official_24crop',
                             'seed': 3407})
    if args.limit_per_task:
        counts = defaultdict(int)
        selected = []
        for row in requests:
            task = row['input']['answer_space']['kind']
            if counts[task] < args.limit_per_task:
                selected.append(row)
                counts[task] += 1
        requests = selected
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in requests))
        (args.output / 'manifest.json').write_text(json.dumps({
            'n_requests': len(requests), 'profiles': ['original_default: 2 crops / 4096 / deterministic option shuffle',
                                                     'official_24crop: 24 crops / 64000 / original option order'],
            'no_ground_truth_used': True, 'checkpoint': str(args.checkpoint.resolve()), 'merged': str(args.model.resolve()),
            'criteria': {'clean_exact': True, 'choice_agreement': .99, 'scalar_root_agreement': .99,
                         'scalar_value_agreement': .95, 'point_root_agreement': .98, 'point_exact_agreement': .95,
                         'point_p95_normalized_l2_max': math.sqrt(2)/27}}, indent=2) + '\n')
    if world > 1:
        dist.barrier()
    legacy = build_unified_model(config, args.checkpoint, trainable=False).to(device)
    original_processor = make_processor(config['base_model'], config['max_crops'])
    release = JevSpatial.from_pretrained(args.model, device=device)
    rows = []
    profile_cache = {}
    with (args.output / f'results-rank{rank:02d}.jsonl').open('w') as stream:
        for i, row in enumerate(requests[rank::world]):
            profile = row['profile']
            if profile not in profile_cache:
                cfg = {**config}
                if profile == 'official_24crop':
                    cfg.update(max_crops=24, max_sequence_length=64000, preserve_option_order=True)
                original_processor.image_processor.max_crops = cfg['max_crops']
                release.processor.image_processor.max_crops = cfg['max_crops']
                profile_cache[profile] = (UnifiedBuilder(original_processor, cfg),
                                          JevSpatial(legacy, original_processor, cfg, device),
                                          JevSpatial(release.model, release.processor,
                                                     {**cfg, 'runtime_dtype': release.config['runtime_dtype']}, device))
            builder, clean_adapter, clean_merged = profile_cache[profile]
            original_processor.image_processor.max_crops = clean_adapter.config['max_crops']
            release.processor.image_processor.max_crops = clean_merged.config['max_crops']
            inp, seed = row['input'], row['seed']
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                before = predict_unified(legacy, builder, inp, device, seed)
            clean = clean_adapter.predict(inp, seed=seed)
            if clean != before:
                raise AssertionError(f"Clean code changed output for {row['id']}")
            after = clean_merged.predict(inp, seed=seed)
            task = inp['answer_space']['kind']
            a, b = before['prediction'], after['prediction']
            distance = float(a != b) if task == 'choice' else abs(a-b) if task == 'scalar' else math.dist(a, b)
            deltas = []
            for level, (x, y) in enumerate(zip(before['logits'], after['logits'])):
                deltas.append(max(abs(u-v) for u, v in zip(x, y)))
                if before['path'][level] != after['path'][level]:
                    break
            result = {'id': row['id'], 'task': task, 'profile': profile, 'clean_exact': True,
                      'same_prediction': a == b, 'output_distance': distance,
                      'max_same_prefix_logit_delta': max(deltas), 'before': before, 'after': after}
            rows.append(result)
            stream.write(json.dumps(result, allow_nan=False) + '\n')
            stream.flush()
            if (i+1) % 50 == 0:
                print(json.dumps({'rank': rank, 'done': i+1}), flush=True)
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        rows = [r for part in gathered for r in part]
    if rank == 0:
        summary = summarize(rows)
        summary['by_profile'] = {p: summarize([r for r in rows if r['profile'] == p])
                                 for p in sorted({r['profile'] for r in rows})}
        summary['overall_criteria_passed'] = summary['passed']
        summary['passed'] = summary['passed'] and all(v['passed'] for v in summary['by_profile'].values())
        (args.output / 'results.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
