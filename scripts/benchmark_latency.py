"""Paired batch-one end-to-end latency on fixed samples from each spatial benchmark."""

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from spatial_jev.batching import make_processor, prepare_batch  # noqa: E402
from spatial_jev.inference import JevSpatial  # noqa: E402
from spatial_jev.model import build_model  # noqa: E402
from evaluate_fast_benchmarks import native_official_predict, native_predict  # noqa: E402

METHODS = ['native', 'three_head_merged', 'shared_head_merged']
BENCHMARKS = ['sat_real', 'cv_bench', 'refspatial_bench', 'where2place',
              'robospatial_poi', 'robospatial_vq', 'vst_numeric_dev']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_requests(n, seed):
    groups = defaultdict(list)
    source_hashes = {}
    for suite in ['official_images_v1', 'robospatial_v1', 'fast_v1']:
        path = ROOT / 'data/benchmarks' / suite / 'requests.jsonl'
        source_hashes[suite] = sha(path)
        for row in map(json.loads, path.read_text().splitlines()):
            if suite == 'fast_v1' and row['benchmark'] != 'vst_numeric_dev':
                continue
            # One option order per SAT question: sample 20 distinct questions.
            if row['benchmark'] == 'sat_real' and row['rotation'] != 0:
                continue
            groups[row['benchmark']].append(row)
    rng = random.Random(seed)
    selected = []
    for benchmark in BENCHMARKS:
        families = defaultdict(list)
        for row in groups[benchmark]:
            families[row['family']].append(row)
        total = sum(map(len, families.values()))
        quotas = {k: n * len(rows) // total for k, rows in families.items()}
        remainder = sorted(families, key=lambda k: (-(n * len(families[k]) % total), k))
        for family in remainder[:n - sum(quotas.values())]:
            quotas[family] += 1
        for family in sorted(families):
            selected.extend(rng.sample(families[family], quotas[family]))
    rng.shuffle(selected)
    return selected, source_hashes


def typed_predict(model, processor, row, config, device):
    inp = row['input']
    task = inp['answer_space']['kind']
    dummy = ({'option_id': inp['answer_space']['options'][0]['id']} if task == 'choice'
             else {'value': 0.0} if task == 'scalar' else {'points': [[0.0, 0.0]]})
    inputs, _, counts, mappings = prepare_batch(
        [{'input': inp, 'target': dummy}], processor, device, config, seed=None)
    value = model(inputs, task, counts)
    prediction = (mappings[0][int(value[0].argmax())] if task == 'choice' else
                  value[0].clamp(max=20).expm1().item() if task == 'scalar' else
                  value[0].cpu().tolist())
    return {'prediction': prediction, 'input_tokens': inputs['input_ids'].shape[1]}


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['benchmark'], row['id'], row['method'])].append(row)
    samples = []
    for (benchmark, key, method), repeats in grouped.items():
        samples.append({'benchmark': benchmark, 'id': key, 'method': method,
                        'milliseconds': statistics.median(r['milliseconds'] for r in repeats),
                        'parse_failures': sum(r['output'].get('prediction') is None for r in repeats),
                        'output_tokens': repeats[0]['output'].get('output_tokens'),
                        'generation_truncated': any(r['output'].get('generation_truncated', False)
                                                    for r in repeats)})
    result = {}
    for benchmark in [*BENCHMARKS, 'overall']:
        result[benchmark] = {}
        for method in METHODS:
            rows = [r for r in samples if r['method'] == method
                    and (benchmark == 'overall' or r['benchmark'] == benchmark)]
            values = [r['milliseconds'] for r in rows]
            tokens = [r['output_tokens'] for r in rows if r['output_tokens'] is not None]
            result[benchmark][method] = {
                'n': len(values), 'mean_ms': statistics.mean(values),
                'p50_ms': float(np.quantile(values, .5)), 'p95_ms': float(np.quantile(values, .95)),
                'min_ms': min(values), 'max_ms': max(values),
                'parse_failure_requests': sum(r['parse_failures'] > 0 for r in rows),
                'truncated_requests': sum(r['generation_truncated'] for r in rows),
                'mean_output_tokens': statistics.mean(tokens) if tokens else None,
            }
        native = result[benchmark]['native']['mean_ms']
        for method in METHODS:
            result[benchmark][method]['speedup_vs_native'] = native / result[benchmark][method]['mean_ms']
    return {'by_benchmark': result, 'per_sample': samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--base-model', type=Path, default=ROOT / 'models/allenai/Molmo2-ER')
    parser.add_argument('--three-head-checkpoint', type=Path,
                        default=ROOT / 'outputs/pilot_v0/20260923T161433Z/full/checkpoint-001000')
    parser.add_argument('--merged-model', type=Path,
                        default=ROOT / 'releases/publication-v0.1.0/huggingface')
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        parser.error('samples and repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    requests, hashes = select_requests(args.samples, args.seed)
    (args.output / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in requests))
    config = json.loads((args.three_head_checkpoint / 'config.json').read_text())
    configs = {
        crops: {**config, 'max_crops': crops, 'max_sequence_length': length,
                'preserve_option_order': True, 'native_point_policy': 'single',
                'robospatial_vq_prompt': 'source'}
        for crops, length in [(2, 4096), (24, 64000)]}
    meta = {
        'samples_per_benchmark': args.samples, 'repeats': args.repeats, 'seed': args.seed,
        'sample_selection': 'fixed random seed, proportional task-family strata, distinct SAT questions',
        'source_sha256': hashes, 'request_sha256': sha(args.output / 'requests.jsonl'),
        'benchmark_counts': dict(Counter(r['benchmark'] for r in requests)),
        'families': {b: dict(Counter(r['family'] for r in requests if r['benchmark'] == b)) for b in BENCHMARKS},
        'device': args.device, 'gpu': torch.cuda.get_device_name(device),
        'python': platform.python_version(), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'cuda': torch.version.cuda,
        'batch_size': 1, 'cpu_threads': 4, 'precision': 'BF16 backbone + FP32 task heads',
        'lora': 'both adapted methods merged; three-head merge performed once before timing',
        'image_profile': '24 crops / 64000 tokens for six image benchmarks; VST dev 2 / 4096',
        'native_point_policy': 'one point for every pointing benchmark',
        'native_max_new_tokens': {'image_benchmarks': 256, 'vst_numeric_dev': 128},
        'timing': 'synchronized wall time: image IO, CPU preprocessing, H2D, inference, decoding/parsing; excludes model load and result-file IO',
        'cache': 'warm OS file cache; no processed-image cache or KV reuse across requests',
        'execution': 'three models resident on the same GPU, requests serial; rotated method order across repeats',
        'warmup': 'one request per benchmark and method, excluded',
        'aggregation': 'median over repeats per request, then mean/P50/P95 over 20 requests; equal benchmark weighting overall',
        'base_revision': config['base_revision'],
        'three_head_step': json.loads((args.three_head_checkpoint / 'checkpoint.json').read_text())['step'],
        'three_head_adapter_sha256': sha(args.three_head_checkpoint / 'adapter/adapter_model.safetensors'),
        'three_head_heads_sha256': sha(args.three_head_checkpoint / 'heads.safetensors'),
        'shared_release_manifest_sha256': sha(args.merged_model / 'manifest.json'),
        'script_sha256': sha(__file__),
    }
    (args.output / 'config.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('Loading three models on one GPU', flush=True)
    native = transformers.AutoModelForImageTextToText.from_pretrained(
        args.base_model, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    typed = build_model(args.base_model, config, args.three_head_checkpoint, trainable=False).to(device)
    typed.backbone = typed.backbone.merge_and_unload(safe_merge=True)
    typed.eval()
    shared = JevSpatial.from_pretrained(args.merged_model, device=device, preserve_option_order=True)
    processor = make_processor(args.base_model, 24)
    shared_runners = {crops: JevSpatial(shared.model, shared.processor,
                     {**shared.config, **{k: v for k, v in cfg.items()
                                         if k in ('max_crops', 'max_sequence_length', 'preserve_option_order')}}, device)
                      for crops, cfg in configs.items()}
    for path in {m['uri'] for r in requests for m in r['input']['media']}:
        Path(path).read_bytes()

    def configure(row):
        crops = 2 if row['benchmark'] == 'vst_numeric_dev' else 24
        processor.image_processor.max_crops = crops
        shared.processor.image_processor.max_crops = crops
        return configs[crops], shared_runners[crops]

    def predict(method, row, cfg, runner):
        if method == 'native':
            if row['benchmark'] == 'vst_numeric_dev':
                return native_predict(native, processor, row['input'], cfg, device)
            return native_official_predict(native, processor, row, cfg, device)
        if method == 'three_head_merged':
            return typed_predict(typed, processor, row, cfg, device)
        return runner.predict(row['input'], seed=args.seed)

    records = []
    order_rng = random.Random(args.seed + 1)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for benchmark in BENCHMARKS:
            row = next(r for r in requests if r['benchmark'] == benchmark)
            cfg, runner = configure(row)
            for method in METHODS:
                predict(method, row, cfg, runner)
            print(json.dumps({'warmup': benchmark}), flush=True)
        torch.cuda.synchronize(device)
        print('Warmup complete; measuring', flush=True)
        with (args.output / 'measurements.jsonl').open('w') as stream:
            for i, row in enumerate(requests):
                cfg, runner = configure(row)
                order = list(METHODS)
                order_rng.shuffle(order)
                for repeat in range(args.repeats):
                    rotated = order[repeat % 3:] + order[:repeat % 3]
                    for position, method in enumerate(rotated):
                        torch.cuda.synchronize(device)
                        begin = time.perf_counter()
                        out = predict(method, row, cfg, runner)
                        torch.cuda.synchronize(device)
                        elapsed = (time.perf_counter() - begin) * 1000
                        result = {'id': row['id'], 'benchmark': row['benchmark'], 'family': row['family'],
                                  'task': row['input']['answer_space']['kind'], 'method': method,
                                  'repeat': repeat, 'order_position': position, 'milliseconds': elapsed,
                                  'output': out}
                        records.append(result)
                        stream.write(json.dumps(result, allow_nan=False) + '\n')
                        stream.flush()
                if (i + 1) % 10 == 0:
                    progress = {'requests_complete': i + 1, 'total': len(requests)}
                    (args.output / 'progress.json').write_text(json.dumps(progress) + '\n')
                    print(json.dumps(progress), flush=True)
    summary = {'config': meta, **summarize(records)}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['by_benchmark'], indent=2), flush=True)


if __name__ == '__main__':
    main()
