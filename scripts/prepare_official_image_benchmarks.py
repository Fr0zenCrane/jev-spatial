"""Freeze full SAT, RefSpatial, CV-Bench and Where2Place test requests and separate labels."""

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/benchmarks/official_images_v1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    images = args.output / 'images'
    images.mkdir()
    fast = ROOT / 'data/benchmarks/fast_v1'
    requests = [json.loads(s) for s in (fast / 'requests.jsonl').read_text().splitlines()
                if json.loads(s)['benchmark'] != 'vst_numeric_dev']
    ids = {r['id'] for r in requests}
    references = [json.loads(s) for s in (fast / 'references.jsonl').read_text().splitlines()
                  if json.loads(s)['id'] in ids]
    sources = {str(fast / 'summary.json'): digest((fast / 'summary.json').read_bytes())}

    def media(blob):
        sha = digest(blob)
        with Image.open(io.BytesIO(blob)) as image:
            width, height = image.size
            rgb = image.convert('RGB')
            rgb_sha = digest(str(rgb.size).encode() + rgb.tobytes())
        path = images / (sha + '.img')
        if not path.exists():
            path.write_bytes(blob)
        return {'kind': 'image', 'uri': str(path.resolve()), 'sha256': sha,
                'rgb_sha256': rgb_sha, 'width': width, 'height': height}

    # Preserve all original choice orderings. No target-dependent prompt content.
    for row in requests:
        inp = row['input']
        if inp['answer_space']['kind'] == 'choice':
            options = inp['answer_space']['options']
            row['native_prompt'] = inp['question'] + ' Choose from the following options:\n' + '\n'.join(
                f"({chr(65+i)}) {o['text']}" for i, o in enumerate(options)) + '\nAnswer with only the option letter.'
        else:
            row['native_prompt'] = inp['question']
    for filename in ['test_2d.parquet', 'test_3d.parquet']:
        path = ROOT / 'data/raw/nyu-visionx/CV-Bench' / filename
        sources[str(path)] = digest(path.read_bytes())
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                options = [{'id': f'option_{i}', 'text': choice} for i, choice in enumerate(row['choices'])]
                index = ord(row['answer'].strip('()')) - 65
                assert 0 <= index < len(options)
                key = f"cv_bench:{row['idx']}"
                requests.append({'id': key, 'benchmark': 'cv_bench', 'group_id': key,
                                 'family': row['task'], 'source': row['source'], 'dimension': row['type'],
                                 'native_prompt': row['prompt'] + '\nAnswer with only the option letter.',
                                 'input': {'media': [media(row['image']['bytes'])], 'question': row['question'],
                                           'answer_space': {'kind': 'choice', 'options': options}}})
                references.append({'id': key, 'target': {'option_id': options[index]['id']}})
    directory = ROOT / 'data/raw/wentao-yuan/where2place'
    questions = directory / 'point_questions.jsonl'
    sources[str(questions)] = digest(questions.read_bytes())
    for row in map(json.loads, questions.read_text().splitlines()):
        key = f"where2place:{row['question_id']}"
        requests.append({'id': key, 'benchmark': 'where2place', 'group_id': key,
                         'family': row['category'], 'native_prompt': row['text'],
                         'input': {'media': [media((directory / 'images' / row['image']).read_bytes())],
                                   'question': row['text'].split('Your answer should be formatted')[0].strip(),
                                   'answer_space': {'kind': 'point', 'coordinate_system': 'normalized_xy', 'num_points': 1}}})
        mask = directory / 'masks' / f"{row['question_id']:02d}.jpg"
        with Image.open(mask) as im:
            assert im.size == (640, 480) and im.mode == 'L'
        references.append({'id': key, 'target': {'mask_path': str(mask.resolve())}})
    counts = Counter(r['benchmark'] for r in requests)
    assert counts == {'sat_real': 300, 'refspatial_bench': 200, 'cv_bench': 2638, 'where2place': 100}
    assert len({r['id'] for r in requests}) == len(requests)
    # Audit only; never drop test cases based on training overlap.
    train_sha, train_rgb = set(), set()
    for path in [ROOT / 'data/processed/mixed_v2/20260923T203244Z/train.jsonl',
                 ROOT / 'data/processed/pilot_v0/20260923T160355Z/train.jsonl']:
        for line in path.open():
            for image in json.loads(line)['input']['media']:
                train_sha.add(image['sha256'])
                if 'rgb_sha256' in image:
                    train_rgb.add(image['rgb_sha256'])
    overlap = {}
    cache = {}
    for row in requests:
        matches = []
        for item in row['input']['media']:
            sha = item['sha256']
            if sha not in cache:
                with Image.open(item['uri']) as image:
                    rgb = image.convert('RGB')
                    cache[sha] = digest(str(rgb.size).encode() + rgb.tobytes())
            if sha in train_sha or cache[sha] in train_rgb:
                matches.append(sha)
        if matches:
            overlap[row['id']] = matches
    for name, rows in [('requests', requests), ('references', references)]:
        (args.output / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    summary = {'suite': 'official_images_v1', 'requests_per_benchmark': dict(counts),
               'n_requests': len(requests), 'source_sha256': sources,
               'requests_sha256': digest((args.output / 'requests.jsonl').read_bytes()),
               'references_sha256': digest((args.output / 'references.jsonl').read_bytes()),
               'training_media_overlap_requests': overlap,
               'overlap_scope': 'byte and available decoded RGB hashes; not a physical-scene or base-model-membership audit',
               'scope': 'complete official test sets and scoring; local model adapters, not certified paper-exact or submitted leaderboard results',
               'point_protocol': 'classifier singleton; native point set; official mean mask value over predicted points',
               'cv_aggregation': '0.25 ADE20K + 0.25 COCO + 0.5 Omni3D',
               'where2place_mask': 'official JPEG grayscale / 255, NOT thresholded mask',
               'no_official_numeric_score': 'VST internal dev excluded; VSI video and count/area interfaces not yet implemented'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
