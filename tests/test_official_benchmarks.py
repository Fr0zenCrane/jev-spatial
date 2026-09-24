import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spatial_jev.official_benchmarks import (cv_source_score, official_mask_score,
                                            parse_choice, parse_points)


def test_official_cv_uses_source_weights_not_micro_average():
    rows = [{'source': 'ADE20K', 'score': 1}] * 10 + [
        {'source': 'COCO', 'score': 0}, {'source': 'Omni3D', 'score': 0}]
    score, _ = cv_source_score(rows)
    assert score == .25
    assert cv_source_score(rows[:10])[0] is None


def test_letter_parser_does_not_pick_among_ambiguous_candidates():
    options = [{'id': 'x', 'text': 'left'}, {'id': 'y', 'text': 'right'}]
    assert parse_choice('<answer>(B)</answer>', options) == 'y'
    assert parse_choice('Answer: A', options) == 'x'
    with pytest.raises(ValueError):
        parse_choice('Could be A or B', options)


def test_molmo2_html_multiple_points_and_legacy_coordinates():
    points, _ = parse_points('<points coords="1 1 250 500 2 750 125">cup</points>', 640, 480)
    assert points == [[.25, .5], [.75, .125]]
    assert parse_points('<point x="25.0" y="50.0">cup</point>', 640, 480)[0] == [[.25, .5]]
    assert parse_points('[(0.25, 0.5), (480, 60)]', 640, 480)[0] == [[.25, .5], [.75, .125]]


def test_where2place_preserves_grayscale_and_invalid_point_denominator(tmp_path):
    path = tmp_path / 'mask.png'
    Image.fromarray(np.array([[0, 128], [255, 0]], dtype=np.uint8)).save(path)
    assert official_mask_score([[.75, .25]], path, binary=False) == 128 / 255
    assert official_mask_score([[.75, .25]], path, binary=True) == 1
    assert official_mask_score([[.25, .75], [1.0, .75]], path, binary=False) == .5
    assert official_mask_score([], path) == 0


def test_refspatial_matches_pinned_official_scorer(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    source = root / 'artifacts/research/official-eval-20260924/refspatial_summarize_acc.py'
    if not source.exists():
        pytest.skip('Pinned external scorer snapshot not downloaded')
    spec = importlib.util.spec_from_file_location('refspatial_official', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / 'mask.png'
    Image.fromarray(np.array([[0, 128], [255, 0]], dtype=np.uint8)).save(path)
    points = [[.25, .75], [1.0, .75], [.75, .25]]
    answers = [{'question_id': 0, 'mask_path': str(path), 'text': str([tuple(p) for p in points])}]
    module.compute_accuracy(answers, 'location', lambda text, w, h: module.text2pts(text, w, h, is_absolute=False))
    assert official_mask_score(points, path) == answers[0]['accuracy']
