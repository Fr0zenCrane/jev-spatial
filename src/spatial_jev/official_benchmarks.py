"""Label-independent output adapters and official image benchmark aggregations."""

import math
import re
from collections import defaultdict

import numpy as np
from PIL import Image

from .benchmark_metrics import summarize_benchmarks


def parse_choice(text, options):
    text = text.strip()
    tagged = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, flags=re.S | re.I)
    if len(tagged) == 1:
        text = tagged[0].strip()
    text = re.sub(r'^(?:the\s+)?(?:final\s+)?(?:answer|option)(?:\s+is)?\s*:?\s*', '', text, flags=re.I)
    match = re.fullmatch(r'\(?([A-Z])\)?[.]?', text, flags=re.I)
    if match:
        index = ord(match.group(1).upper()) - 65
        if 0 <= index < len(options):
            return options[index]['id']
        raise ValueError('choice_letter_out_of_range')
    found = [o['id'] for o in options if o['text'].strip().rstrip('.').casefold() == text.rstrip('.').casefold()]
    if len(found) == 1:
        return found[0]
    raise ValueError('not_an_unambiguous_official_choice')


def parse_points(text, width, height):
    """Decode Molmo2 HTML (1000), legacy XML (100), or benchmark tuple coordinates."""
    points = []
    groups = re.findall(r'<(?:points|tracks)[^>]*\bcoords="([0-9\t:;, .]+)"', text)
    if groups:
        for group in groups:
            frames = re.finditer(r'(?:^|\t|:|,|;)([0-9.]+) ([0-9. ]+)', group)
            for frame in frames:
                if float(frame.group(1)) != 1:
                    raise ValueError('unexpected_image_index')
                for point in re.finditer(r'([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})', frame.group(2)):
                    points.append([float(point.group(2)) / 1000, float(point.group(3)) / 1000])
        mode = 'molmo2_html_1000'
    elif re.search(r'<points?\b', text):
        for tag in re.findall(r'<points?\b[^>]*>', text):
            attrs = dict(re.findall(r'\b([xy]\d*)="(-?\d+(?:\.\d+)?)"', tag))
            for xname, value in attrs.items():
                if xname.startswith('x') and 'y' + xname[1:] in attrs:
                    points.append([float(value) / 100, float(attrs['y' + xname[1:]]) / 100])
        mode = 'legacy_molmo_xml_100'
    else:
        number = r'[-+]?\d+\.?\d*'
        for xtext, ytext in re.findall(rf'\(\s*({number})\s*,\s*({number})\s*\)', text):
            x, y = float(xtext), float(ytext)
            if '.' not in xtext and '.' not in ytext:
                x, y = x / width, y / height
            points.append([x, y])
        mode = 'benchmark_tuples'
    if not points or any(not math.isfinite(v) for point in points for v in point):
        raise ValueError('no_parseable_points')
    return points, mode


def official_mask_score(points, mask_path, binary=True):
    """Match official integer conversion and invalid-point denominator; no clipping."""
    mask = np.asarray(Image.open(mask_path)) / 255.0
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if binary:
        mask = mask > 0
    height, width = mask.shape
    if not points:
        return 0.0
    values = []
    for x, y in points:
        if not math.isfinite(x) or not math.isfinite(y):
            values.append(0.0)
            continue
        px, py = int(x * width), int(y * height)
        values.append(float(mask[py, px]) if 0 <= px < width and 0 <= py < height else 0.0)
    return float(np.mean(values))


def cv_source_score(rows):
    sources = {name: float(np.mean([r['score'] for r in rows if r['source'] == name]))
               for name in {r['source'] for r in rows}}
    complete = set(sources) == {'ADE20K', 'COCO', 'Omni3D'}
    return ((sources['ADE20K'] + sources['COCO']) / 4 + sources['Omni3D'] / 2
            if complete else None), sources


def summarize_official(requests, references, predictions):
    expected = {r['id'] for r in requests}
    if set(references) != expected or set(predictions) != expected:
        raise ValueError('Scoring requires every requested prediction and reference, including failures')
    summary, grouped = {}, defaultdict(list)
    sat = [r for r in requests if r['benchmark'] == 'sat_real']
    if sat:
        ids = {r['id'] for r in sat}
        summary.update(summarize_benchmarks(sat, {k: v for k, v in references.items() if k in ids},
                                            {k: v for k, v in predictions.items() if k in ids}))
    for request in requests:
        name, key = request['benchmark'], request['id']
        if name == 'sat_real':
            continue
        prediction, target = predictions[key], references[key]['target']
        row = {'id': key, 'family': request['family'], 'source': request.get('source')}
        if name == 'cv_bench':
            row.update(valid=prediction.get('prediction') is not None,
                       score=float(prediction.get('prediction') == target['option_id']))
        elif name in ('refspatial_bench', 'where2place'):
            xy = prediction.get('prediction')
            points = prediction.get('points', [xy] if xy is not None else [])
            row.update(valid=bool(points), point_count=len(points),
                       score=official_mask_score(points, target['mask_path'], binary=name == 'refspatial_bench'))
        else:
            raise ValueError(f'Unsupported official benchmark {name}')
        grouped[name].append(row)
    for name, rows in grouped.items():
        result = {'n_requests': len(rows), 'valid_rate': float(np.mean([r['valid'] for r in rows])),
                  'score': float(np.mean([r['score'] for r in rows])),
                  'by_family': {f: {'n': sum(r['family'] == f for r in rows),
                                    'score': float(np.mean([r['score'] for r in rows if r['family'] == f]))}
                                for f in sorted({r['family'] for r in rows})}}
        if name == 'cv_bench':
            result['micro_accuracy'] = result['score']
            result['score'], result['by_source'] = cv_source_score(rows)
            result['metric'] = 'official 0.25 ADE20K + 0.25 COCO + 0.5 Omni3D accuracy'
        else:
            result['mean_predicted_points'] = float(np.mean([r['point_count'] for r in rows]))
            result['metric'] = ('official binary mask mean point accuracy' if name == 'refspatial_bench'
                                else 'official JPEG mask value / 255 averaged over points and questions')
        summary[name] = result
    return summary
