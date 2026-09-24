import numpy as np
import pytest
from PIL import Image

from spatial_jev.benchmark_metrics import point_mask_score, summarize_benchmarks
from spatial_jev.unified import UnifiedBuilder


def test_distinct_valid_points_both_score_one(tmp_path):
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[2:8, 3:18] = 255
    path = tmp_path / "mask.png"
    Image.fromarray(mask).save(path)
    assert point_mask_score([[.2, .3]], path) == 1
    assert point_mask_score([[.8, .7]], path) == 1
    assert point_mask_score([[.2, .3], [0, 0]], path) == .5
    assert point_mask_score([[1, .5]], path) == 0  # Never clamp an invalid point onto the object.
    assert point_mask_score(None, path) == 0


def test_sat_both_orders_and_failures_remain_in_denominator():
    requests = [{"id": str(i), "benchmark": "sat_real", "family": "motion",
                 "group_id": "q0", "rotation": i} for i in range(2)]
    refs = {str(i): {"target": {"option_id": "a"}} for i in range(2)}
    preds = {"0": {"prediction": "a"}, "1": {"prediction": None}}
    score = summarize_benchmarks(requests, refs, preds)["sat_real"]
    assert score["score"] == .5 and score["both_orders_correct"] == 0
    with pytest.raises(ValueError):
        summarize_benchmarks(requests, refs, {"0": preds["0"]})


def test_scalar_parse_failure_has_no_fake_finite_mae():
    requests = [{"id": "x", "benchmark": "vst_numeric_dev", "family": "height", "group_id": "g"}]
    score = summarize_benchmarks(requests, {"x": {"target": {"value": 1}}},
                                 {"x": {"prediction": None}})["vst_numeric_dev"]
    assert score["mae_m_all"] is None and score["within_10cm"] == 0
    assert score["absrel_positive_valid"] is None


def test_explicit_benchmark_order_is_preserved_for_all_seeds():
    builder = object.__new__(UnifiedBuilder)
    builder.config = {"preserve_option_order": True}
    builder.render = lambda text, image_count: text
    inp = {"media": [{}], "question": "Where?", "answer_space": {"kind": "choice", "options": [
        {"id": "b", "text": "right"}, {"id": "a", "text": "left"}]}}
    for seed in range(10):
        text, count, mapping = builder.initial(inp, seed)
        assert count == 2 and mapping == ["b", "a"]
        assert "1. right\n2. left" in text
