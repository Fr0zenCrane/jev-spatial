"""Prepare a code-free Hugging Face model directory from the verified merged export."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_CARD_METADATA = ('---\nlicense: apache-2.0\nbase_model: allenai/Molmo2-ER\n'
                       'language: [en, zh]\n'
                       'tags: [spatial-reasoning, multimodal, classification, pointing]\n---\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def strip_auto_map(value):
    if isinstance(value, dict):
        return {k: strip_auto_map(v) for k, v in value.items() if k != 'auto_map'}
    if isinstance(value, list):
        return [strip_auto_map(v) for v in value]
    return value


def stage_model(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    original = json.loads((source / 'manifest.json').read_text())
    if original['merge']['adapter_runtime_required']:
        raise ValueError('Expected merged weights')
    members = []
    for relative, expected in original['files'].items():
        path = source / relative
        include = relative in ('jev_config.json', 'classifier.safetensors')
        include |= relative.startswith('backbone/') and path.suffix in ('.json', '.safetensors')
        include |= relative.startswith('processor/') and path.suffix in ('.json', '.txt', '.jinja')
        if not include:
            continue
        if not path.resolve().is_relative_to(source) or path.is_symlink():
            raise ValueError(f'Invalid model member: {relative}')
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError(f'Checksum mismatch: {relative}')
        members.append((relative, path, expected))
    output.mkdir(parents=True)
    hashes = {}
    for relative, path, expected in members:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.safetensors':
            try:
                os.link(path, target)  # Immutable local tensors; no extra 8.9 GB copy.
            except OSError:
                shutil.copy2(path, target)
            hashes[relative] = expected
        elif path.suffix == '.json':
            value = strip_auto_map(json.loads(path.read_text()))
            target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        else:
            shutil.copy2(path, target)
    for name in ['README.md', 'README-zh.md', 'LICENSE', 'NOTICE']:
        shutil.copy2(ROOT / name, output / name)
    readme = output / 'README.md'
    if not readme.read_text().startswith('---\n'):
        readme.write_text(MODEL_CARD_METADATA + readme.read_text())
    (output / '.gitattributes').write_text('*.safetensors filter=lfs diff=lfs merge=lfs -text\n')
    removed = {'files', 'standalone_runtime', 'standalone_validation_passed',
               'validation_summary_sha256', 'code_license', 'checkpoint_contributions_license'}
    manifest = {k: v for k, v in original.items() if k not in removed}
    manifest.update(license='Apache-2.0', runtime_source='installed spatial_jev package',
                    contains_python_code=False)
    manifest['implementation_files'] = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted((ROOT / 'src/spatial_jev').rglob('*.py'))
        if '__pycache__' not in p.parts}
    report = source / 'validation/parity.json'
    if report.exists():
        manifest['merge_verification'] = json.loads(report.read_text())
    manifest['files'] = {
        str(p.relative_to(output)): hashes.get(str(p.relative_to(output))) or
        {'bytes': p.stat().st_size, 'sha256': digest(p)}
        for p in sorted(output.rglob('*')) if p.is_file()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    assert not list(output.rglob('*.py'))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = stage_model(args.model, args.output)
    print(json.dumps({'files': len(result['files']) + 1, 'license': result['license'],
                      'contains_python_code': False}))


if __name__ == '__main__':
    main()
