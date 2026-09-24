"""Prepare the complete pinned RoboSpatial-Home RGB benchmark, with isolated references."""

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/benchmarks/robospatial_v1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    images, masks = args.output / 'images', args.output / 'masks'
    images.mkdir()
    masks.mkdir()
    manifest = json.loads((ROOT / 'data/manifests/robospatial_eval_v1_assets.json').read_text())['sources'][0]
    source = ROOT / 'data/raw' / manifest['repo']
    verified = {}
    for f in manifest['files']:
        blob = (source / f['path']).read_bytes()
        assert len(blob) == f['bytes']
        if f['sha256']:
            assert sha(blob) == f['sha256']
        verified[f['path']] = sha(blob)
    requests, refs = [], []
    train_sha, train_rgb = set(), set()
    for directory in ['mixed_v2/20260923T203244Z', 'pilot_v0/20260923T160355Z']:
        for line in (ROOT / 'data/processed' / directory / 'train.jsonl').open():
            for media in json.loads(line)['input']['media']:
                train_sha.add(media['sha256'])
                if 'rgb_sha256' in media:
                    train_rgb.add(media['rgb_sha256'])
    overlaps = []
    for split in ['configuration', 'compatibility', 'context']:
        rows = pq.read_table(source / 'data' / f'{split}-00000-of-00001.parquet').to_pylist()
        for index, row in enumerate(rows):
            assert row['category'] == split
            blob = row['img']['bytes']
            image_sha = sha(blob)
            image_path = images / (image_sha + '.img')
            image_path.write_bytes(blob)
            with Image.open(io.BytesIO(blob)) as image:
                width, height = image.size
                rgb = image.convert('RGB')
                rgb_sha = sha(str(rgb.size).encode() + rgb.tobytes())
            media = {'kind': 'image', 'uri': str(image_path.resolve()), 'sha256': image_sha,
                     'rgb_sha256': rgb_sha, 'width': width, 'height': height}
            key = f'robospatial:{split}:{index}'
            if image_sha in train_sha or rgb_sha in train_rgb:
                overlaps.append(key)
            if split == 'context':
                benchmark = 'robospatial_poi'
                question = row['question'].split('Your answer should be formatted')[0].strip()
                space = {'kind': 'point', 'coordinate_system': 'normalized_xy', 'num_points': 1}
                mask = row['mask']['bytes']
                mask_path = masks / f'mask_context_{index}.png'
                with Image.open(io.BytesIO(mask)) as image:
                    image.save(mask_path)
                target = {'mask_path': str(mask_path.resolve()), 'source_answer': row['answer']}
            else:
                benchmark = 'robospatial_vq'
                question = row['question']
                space = {'kind': 'choice', 'options': [{'id': 'yes', 'text': 'Yes'}, {'id': 'no', 'text': 'No'}]}
                answer = row['answer'].strip().lower()
                assert answer in ['yes', 'no']
                target = {'option_id': answer, 'source_answer': row['answer']}
            requests.append({'id': key, 'benchmark': benchmark, 'group_id': key, 'family': split,
                             'input': {'media': [media], 'question': question, 'answer_space': space},
                             'native_prompt': question})
            refs.append({'id': key, 'source_question': row['question'], 'source_image': row['img']['path'],
                         'target': target})
    counts = dict(Counter(r['family'] for r in requests))
    assert counts == {'configuration': 123, 'compatibility': 105, 'context': 122}
    for name, rows in [('requests', requests), ('references', refs)]:
        (args.output / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    summary = {'benchmark': 'RoboSpatial-Home', 'dataset_revision': manifest['revision'],
               'counts': counts, 'n_requests': len(requests), 'modality': 'RGB only; depth excluded for all models',
               'source_sha256': verified, 'requests_sha256': sha((args.output / 'requests.jsonl').read_bytes()),
               'references_sha256': sha((args.output / 'references.jsonl').read_bytes()),
               'training_image_overlap_requests': overlaps,
               'overlap_scope': 'byte and available decoded RGB hashes; not scene/base-membership guarantee',
               'current_scorer_revision': 'c0095be6ca2d012086b3c141eccac4c879865cd4',
               'legacy_scorer_revision': 'e1853e2819f835d9decb54f9b89707453af9f833',
               'main_metric': 'Poi: any of first 2 points in mask; VQ: mean of configuration and compatibility accuracies',
               'version_note': 'current main annotations include May 13 yes/no correction; legacy scorer diagnostic is not a certified Molmo2-ER paper reproduction'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
