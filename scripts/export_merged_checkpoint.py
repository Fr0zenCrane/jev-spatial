"""Export a portable, adapter-free Jev-Spatial backbone, classifier and processor."""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from spatial_jev.batching import make_processor  # noqa: E402
from spatial_jev.unified import build_unified_model  # noqa: E402


def file_sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def scrub_local_paths(value):
    if isinstance(value, dict):
        return {k: scrub_local_paths(v) for k, v in value.items() if k not in ['_name_or_path', 'name_or_path']}
    if isinstance(value, list):
        return [scrub_local_paths(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-model', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16',
                        help='Promote the effective BF16 base before merge when exporting FP32')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new export directory; checkpoint exports are immutable')
    checkpoint = args.checkpoint.resolve()
    config = json.loads((checkpoint / 'config.json').read_text())
    if args.base_model:
        config['base_model'] = str(args.base_model.resolve())
    base = Path(config['base_model'])
    marker = json.loads((checkpoint / 'checkpoint.json').read_text())
    torch.set_num_threads(8)
    torch.manual_seed(config['seed'])
    model = build_unified_model(config, checkpoint, trainable=False).to(args.device)
    before_modules = sum(hasattr(m, 'lora_A') for m in model.backbone.modules())
    if before_modules != 144:
        raise ValueError(f'Expected 144 adapted linear layers, found {before_modules}')
    print('Loaded checkpoint; merging', before_modules, 'LoRA layers', flush=True)
    if args.dtype == 'float32':
        # Promote, never reload a different FP32 base: preserve the actual training weights.
        model.backbone.float()
    merged = model.backbone.merge_and_unload(safe_merge=True)
    if any('lora_' in name for name in merged.state_dict()):
        raise ValueError('Unmerged adapter tensor remains')
    merged.eval().requires_grad_(False)
    args.output.mkdir(parents=True)
    model_dir, processor_dir = args.output / 'backbone', args.output / 'processor'
    merged.config.auto_map = {'AutoConfig': 'configuration_molmo2.Molmo2Config',
                              'AutoModel': 'modeling_molmo2.Molmo2Model'}
    merged.config._name_or_path = 'allenai/Molmo2-ER'
    merged.register_for_auto_class('AutoModel')
    merged.save_pretrained(model_dir, safe_serialization=True, max_shard_size='3GB')
    # Ensure the native class and all processor helpers are packaged without hub access.
    for source in base.glob('*.py'):
        shutil.copy2(source, model_dir / source.name)
    processor = make_processor(base, config['max_crops'])
    processor.save_pretrained(processor_dir)
    for source in base.glob('*.py'):
        shutil.copy2(source, processor_dir / source.name)
    shutil.copy2(checkpoint / 'classifier.safetensors', args.output / 'classifier.safetensors')
    keys = ['schema_version', 'seed', 'max_choices', 'max_images', 'max_crops', 'max_sequence_length',
            'architecture', 'point_variant', 'point_depth', 'scalar_codebook']
    public = {key: config[key] for key in keys}
    public.update(format='jev-spatial-merged/v1', preserve_option_order=config.get('preserve_option_order', False),
                  backbone_storage_dtype=args.dtype, runtime_dtype=args.dtype, classifier_dtype='float32',
                  base_model='allenai/Molmo2-ER', base_revision=config['base_revision'],
                  release_name='Jev-Spatial v0.1.0', calibrated=False)
    (args.output / 'jev_config.json').write_text(json.dumps(public, indent=2) + '\n')
    for path in args.output.rglob('*.json'):
        data = scrub_local_paths(json.loads(path.read_text()))
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    manifest = {'format': public['format'], 'created_at': datetime.now(timezone.utc).isoformat(),
                'checkpoint_step': marker['step'], 'initialization_steps': 300,
                'base': {'repo': public['base_model'], 'revision': config['base_revision']},
                'source_adapter_sha256': file_sha(checkpoint / 'adapter/adapter_model.safetensors'),
                'source_classifier_sha256': file_sha(checkpoint / 'classifier.safetensors'),
                'source_training_config_sha256': file_sha(checkpoint / 'config.json'),
                'merge': {'method': 'PEFT merge_and_unload(safe_merge=True)', 'adapted_layers': before_modules,
                          'source_effective_base_dtype': 'bfloat16', 'output_dtype': args.dtype,
                          'adapter_runtime_required': False, 'unused_lm_head_included': False},
                'validation_status': 'pending',
                'files': {str(p.relative_to(args.output)): {'bytes': p.stat().st_size, 'sha256': file_sha(p)}
                          for p in sorted(args.output.rglob('*')) if p.is_file()}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'export': str(args.output.resolve()), 'bytes': sum(v['bytes'] for v in manifest['files'].values()),
                      'layers_merged': before_modules, 'validation': 'pending'}), flush=True)


if __name__ == '__main__':
    main()
