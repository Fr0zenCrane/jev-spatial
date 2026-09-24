"""Stage English/Chinese READMEs, GitHub source and code-free HF weights; never publish."""

import argparse
import json
import re
import shutil
from pathlib import Path

from package_release import MODEL_CARD_METADATA, ROOT, digest, stage_model

SCRIPTS = (
    'download_assets.py', 'download_ranges.py', 'prepare_pilot.py', 'prepare_mixed_v2.py',
    'prepare_fast_benchmarks.py', 'prepare_official_image_benchmarks.py',
    'prepare_robospatial_benchmark.py', 'train_pilot.py', 'train_unified.py',
    'evaluate_fast_benchmarks.py', 'evaluate_generation.py', 'evaluate_typed_checkpoint.py',
    'benchmark_latency.py', 'benchmark_scene_latency.py',
    'predict_unified.py', 'export_merged_checkpoint.py', 'package_release.py',
    'prepare_publication.py', 'verify_release_bundle.py', 'validate_release.py',
)
CREDENTIAL = re.compile(r'(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|'
                        r'github_pat_[A-Za-z0-9_]{20,}|-{5}BEGIN [A-Z ]*PRIVATE KEY-{5})')


def source_paths():
    paths = [ROOT / name for name in
             ['.gitignore', 'README.md', 'README-zh.md', 'LICENSE', 'NOTICE', 'pyproject.toml']]
    paths.extend(ROOT / 'scripts' / name for name in SCRIPTS)
    for tree, suffixes in [('src', {'.py'}), ('configs', {'.json'}),
                           ('tests', {'.py'}), ('examples', {'.json', '.jsonl', '.png'}),
                           ('data/manifests', {'.json'})]:
        paths.extend(p for p in (ROOT / tree).rglob('*') if p.is_file()
                     and p.suffix in suffixes and '__pycache__' not in p.parts
                     and p.name != 'test_repository_contract.py'
                     and not p.name.startswith(('storage_audit_', 'download_files_')))
    for path in sorted(paths):
        if path.is_symlink() or path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f'Unexpected source artifact: {path.relative_to(ROOT)}')
        if path.suffix != '.png' and CREDENTIAL.search(path.read_text()):
            raise ValueError(f'Credential-like content: {path.relative_to(ROOT)}')
        yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new staging directory')
    paths = list(source_paths())
    github, hf = output / 'github', output / 'huggingface'
    github.mkdir(parents=True)
    for path in paths:
        target = github / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    manifest = stage_model(args.model, hf)
    assert (github / 'README.md').read_text() == (hf / 'README.md').read_text().removeprefix(MODEL_CARD_METADATA)
    assert (github / 'README-zh.md').read_bytes() == (hf / 'README-zh.md').read_bytes()
    for folder in (github, hf):
        assert {p.name for p in folder.rglob('*.md')} == {'README.md', 'README-zh.md'}
    report = {'status': 'local_review_only', 'git_commits_created': 0, 'network_writes': 0,
              'readme_languages': ['en', 'zh'], 'license': 'Apache-2.0',
              'hf_contains_python_code': False, 'weight_files_unchanged': True}
    for name, folder in [('github', github), ('huggingface', hf)]:
        files = [p for p in folder.rglob('*') if p.is_file()]
        report[name] = {'files': len(files), 'bytes': sum(p.stat().st_size for p in files)}
    report['source_sha256'] = {str(p.relative_to(github)): digest(p)
                               for p in sorted(github.rglob('*')) if p.is_file()}
    report['merge_validation_status'] = manifest['validation_status']
    (output / 'publication.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
