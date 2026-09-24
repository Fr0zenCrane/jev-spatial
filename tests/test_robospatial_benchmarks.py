from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spatial_jev.robospatial_benchmarks import (official_module, parse_yes_no, prediction_text,
                                              summarize_robospatial, verify_official_aggregate)


@pytest.fixture
def official():
    path = Path(__file__).resolve().parents[1] / 'artifacts/research/robospatial-20260924/evaluation-current.py'
    if not path.exists():
        pytest.skip('Download pinned official RoboSpatial scorer first')
    return official_module()


def test_current_scorer_checks_second_point_and_legacy_checks_first(official, tmp_path):
    mask = tmp_path / 'mask.png'
    Image.fromarray(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 255]], dtype=np.uint8)).save(mask)
    text = '[(0.0, 0.0), (1.0, 1.0)]'
    assert official.evaluate_answer('[(0.9, 0.9)]', text, mask_path=str(mask), num_points_to_match=2)[0]
    assert not official.evaluate_answer('[(0.9, 0.9)]', text, mask_path=str(mask), num_points_to_match=1)[0]
    assert not official_module(True).evaluate_answer('[(0.9, 0.9)]', text, mask_path=str(mask))[0]
    # Preserve the author's rounding/clamping rule rather than the RefSpatial floor rule.
    assert official._normalized_xy_to_mask_indices(2.0, -1.0, 3, 3) == (2, 0)
    assert official._normalized_xy_to_mask_indices(.26, .26, 3, 3) == (1, 1)


def test_yes_no_parsing_and_real_native_text_are_not_gt_dependent():
    assert parse_yes_no(' Yes, it fits.') == 'yes'
    assert parse_yes_no('NO.') == 'no'
    with pytest.raises(ValueError):
        parse_yes_no('It may be possible.')
    assert prediction_text({'benchmark': 'robospatial_vq'}, {'prediction': None, 'raw_answer': 'Possibly.'}) == 'Possibly.'


def test_vqa_macro_average_and_official_full_denominator(official, tmp_path):
    requests, refs, preds = [], {}, {}
    for index, family in enumerate(['configuration', 'configuration', 'compatibility']):
        key = str(index)
        requests.append({'id': key, 'family': family, 'benchmark': 'robospatial_vq'})
        refs[key] = {'source_question': 'q' + key, 'source_image': 'image.jpg',
                     'target': {'source_answer': 'Yes'}}
        preds[key] = {'prediction': 'yes' if family == 'configuration' else None}
    result = summarize_robospatial(requests, refs, preds)['robospatial_vq']
    assert result['score'] == .5
    assert result['micro_accuracy'] == pytest.approx(2/3)
    verified = verify_official_aggregate(requests, refs, preds, tmp_path)
    assert verified['all_scores_match']
    assert verified['checked']['compatibility']['n'] == 1


def test_duplicate_source_records_do_not_overwrite_distinct_predictions(official, tmp_path):
    requests = [{'id': f'robospatial:configuration:{i}', 'family': 'configuration',
                 'benchmark': 'robospatial_vq'} for i in range(2)]
    refs = {r['id']: {'source_question': 'Same question?', 'source_image': '0.jpg',
                       'target': {'source_answer': 'Yes'}} for r in requests}
    preds = {r['id']: {'prediction': 'yes' if i == 0 else 'no'} for i, r in enumerate(requests)}
    result = verify_official_aggregate(requests, refs, preds, tmp_path)
    assert result['checked']['configuration'] == {'n': 2, 'score': .5}


def test_numbered_native_vq_uses_fixed_option_mapping_without_gt():
    request = {'benchmark': 'robospatial_vq'}
    mapped = {'raw_answer': '2', 'prediction': 'no', 'robospatial_binary_format': 'numbered_choices'}
    assert prediction_text(request, mapped) == 'no'
    assert prediction_text(request, {**mapped, 'prediction': None}) == ''
