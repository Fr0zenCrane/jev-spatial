"""RoboSpatial official scoring: current first-two rule and legacy first-point diagnostic."""

import contextlib
import importlib.util
import io
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / 'artifacts/research/robospatial-20260924'


def parse_yes_no(text):
    """The author scorer uses case-insensitive startswith, not an LLM text judge."""
    value = text.strip().lower()
    for answer in ['yes', 'no']:
        if value.startswith(answer):
            return answer
    raise ValueError('not_a_yes_no_prefix')


@lru_cache(maxsize=2)
def official_module(legacy=False):
    filename = 'evaluation-pre-may14.py' if legacy else 'evaluation-current.py'
    spec = importlib.util.spec_from_file_location('robospatial_' + ('legacy' if legacy else 'current'), SNAPSHOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prediction_text(request, prediction):
    if request['benchmark'] == 'robospatial_vq':
        # The native yes/no scorer must see the actual text, including malformed responses.
        if 'raw_answer' in prediction and prediction.get('robospatial_binary_format') != 'numbered_choices':
            return prediction['raw_answer']
        value = prediction.get('prediction')
        return value if value in ('yes', 'no') else ''
    point = prediction.get('prediction')
    points = prediction.get('points', [point] if point is not None else [])
    # Coordinate-format conversion only. No target or mask enters this function.
    return repr([tuple(float(v) for v in xy) for xy in points])


def score_rows(requests, references, predictions):
    expected = {r['id'] for r in requests}
    if expected != set(references) or expected != set(predictions):
        raise ValueError('Every request must retain its prediction and reference, including failures')
    current, legacy = official_module(), official_module(True)
    rows = []
    for request in requests:
        key, family = request['id'], request['family']
        reference, pred = references[key], predictions[key]
        target = reference['target']
        text = prediction_text(request, pred)
        kwargs = {'mask_path': target.get('mask_path'), 'category': family}
        if family == 'context' and not Path(kwargs['mask_path']).is_file():
            raise FileNotFoundError(kwargs['mask_path'])
        correct, _, _, parsable = current.evaluate_answer(target['source_answer'], text, **kwargs, num_points_to_match=2)
        first, _, _, _ = current.evaluate_answer(target['source_answer'], text, **kwargs, num_points_to_match=1)
        old, _, _, _ = legacy.evaluate_answer(target['source_answer'], text, **kwargs)
        point = pred.get('prediction')
        points = pred.get('points', [point] if point is not None and family == 'context' else [])
        rows.append({'id': key, 'family': family, 'correct': bool(correct), 'first_correct': bool(first),
                     'legacy_correct': bool(old), 'parsable': bool(parsable), 'num_points': len(points),
                     'text': text})
    return rows


def summarize_robospatial(requests, references, predictions):
    rows = score_rows(requests, references, predictions)
    by_family = {}
    for family in sorted({r['family'] for r in rows}):
        selected = [r for r in rows if r['family'] == family]
        n = len(selected)
        by_family[family] = {'n': n, 'num_correct': sum(r['correct'] for r in selected),
                             'score': sum(r['correct'] for r in selected) / n,
                             'valid_rate': sum(r['parsable'] for r in selected) / n}
    summary = {}
    if 'context' in by_family:
        selected = [r for r in rows if r['family'] == 'context']
        summary['robospatial_poi'] = {
            **by_family['context'], 'n_requests': len(selected),
            'first_point_score': sum(r['first_correct'] for r in selected) / len(selected),
            'legacy_scorer_score_on_current_annotations': sum(r['legacy_correct'] for r in selected) / len(selected),
            'mean_predicted_points': sum(r['num_points'] for r in selected) / len(selected),
            'metric': 'official current scorer: any of first 2 points hits nonzero grayscale mask; nearest pixel with clamping'}
    groups = {k: v for k, v in by_family.items() if k in ['configuration', 'compatibility']}
    if groups:
        n = sum(v['n'] for v in groups.values())
        summary['robospatial_vq'] = {'n_requests': n,
                                     'score': sum(v['score'] for v in groups.values()) / 2 if len(groups) == 2 else None,
                                     'micro_accuracy': sum(v['num_correct'] for v in groups.values()) / n,
                                     'by_family': groups,
                                     'metric': 'official README VQA average: mean(configuration accuracy, compatibility accuracy)'}
    return summary


def verify_official_aggregate(requests, references, predictions, result_directory):
    """Replay the author's full pre-generated-result evaluator, retaining every example."""
    result_directory = Path(result_directory)
    output = result_directory / 'official-scoring'
    output.mkdir(exist_ok=False)
    metrics = summarize_robospatial(requests, references, predictions)
    checked = {}
    for family in ['configuration', 'compatibility', 'context']:
        group = [r for r in requests if r['family'] == family]
        if not group:
            continue
        gt, answers = [], []
        for row in group:
            reference = references[row['id']]
            # Match the author's PIL downloader naming. The parquet embeds duplicate
            # source basenames/question pairs; each official row still counts separately.
            image_key = f"images/img_{family}_{row['id'].rsplit(':', 1)[-1]}.png"
            gt.append({'question': reference['source_question'], 'img': image_key, 'category': family,
                       'answer': reference['target']['source_answer'], 'mask': reference['target'].get('mask_path')})
            answers.append({'question': reference['source_question'], 'img': image_key,
                            'answer': prediction_text(row, predictions[row['id']])})
        if len({(r['question'], r['img']) for r in gt}) != len(group):
            raise ValueError('Duplicate original question/image keys require an explicit scoring audit')
        (output / f'{family}-predictions.json').write_text(json.dumps(answers, indent=2) + '\n')
        with contextlib.redirect_stdout(io.StringIO()) as log:
            scored = official_module().eval_pregenerated_results(gt, answers, data_dir='', num_points_to_match=2)
        (output / f'{family}.log').write_text(log.getvalue())
        (output / f'{family}.json').write_text(json.dumps(scored, indent=2) + '\n')
        if scored['num_total'] != len(group) or scored['unmatched_entries'] or scored['illformed_questions']:
            raise ValueError('Official scorer changed the complete test denominator')
        expected = metrics['robospatial_poi'] if family == 'context' else metrics['robospatial_vq']['by_family'][family]
        if abs(scored['accuracy'] / 100 - expected['score']) > 1e-12:
            raise ValueError(f'Official aggregation mismatch: {family}')
        checked[family] = {'n': scored['num_total'], 'score': scored['accuracy'] / 100}
    result = {'all_scores_match': True, 'tolerance': 1e-12, 'checked': checked,
              'scorer_revision': 'c0095be6ca2d012086b3c141eccac4c879865cd4'}
    (result_directory / 'official-scorer-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
