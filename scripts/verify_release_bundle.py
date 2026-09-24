"""Verify every published bundle file against its size and SHA-256 manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    root = args.model.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['format'] != 'jev-spatial-merged/v1' or manifest['merge']['adapter_runtime_required']:
        raise ValueError('Not an adapter-free Jev-Spatial bundle')
    for relative, expected in manifest['files'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Invalid or missing bundle member: {relative}')
        if path.stat().st_size != expected['bytes']:
            raise ValueError(f'Size mismatch: {relative}')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != expected['sha256']:
            raise ValueError(f'SHA256 mismatch: {relative}')
    print(json.dumps({'verified_files': len(manifest['files']),
                      'validation_status': manifest['validation_status'],
                      'validation_profiles': manifest.get('validation_profiles', {}),
                      'adapter_runtime_required': False}))


if __name__ == '__main__':
    main()
