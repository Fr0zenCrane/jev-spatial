import math

import numpy as np
import pytest
import torch

from spatial_jev.hierarchy import (POINT_RC, ROOT_BOX, child_box, decode_point, decode_scalar,
                                   point_label, point_path, roi_visible, scalar_path,
                                   visual_token_boxes)
from spatial_jev.unified import (SharedClassifier, apply_roi_mask, causal_visual_mask,
                                 collate_training)

CODEBOOK = {"edges_m": [0, .05, .1, .25, .5, 1, 2, 5, 10, 20, 50, 100, 200],
            "fine_bins": 32, "overflow_representative_m": 200}


def test_user_direction_order_and_nested_centers():
    for i, (row, col) in enumerate(POINT_RC):
        xy = [(col + .5) / 3, (row + .5) / 3]
        assert point_label(xy, ROOT_BOX) == i
        assert decode_point([i]) == pytest.approx(xy)
    for xy in [[0, 0], [1, 1], [.72, .55], [1/3, 2/3]]:
        path = point_path(xy)
        assert len(path) == 3
        decoded = decode_point(path)
        assert math.dist(xy, decoded) <= math.sqrt(2) / 54 + 1e-9


def test_scalar_boundaries_zero_and_explicit_overflow():
    assert scalar_path(0, CODEBOOK) == [0]
    assert scalar_path(201, CODEBOOK) == [13]
    assert scalar_path(200, CODEBOOK) == [12, 31]
    assert scalar_path(.05, CODEBOOK) == [2, 0]
    for x in [.01, .7, 1, 10, 167.82]:
        path = scalar_path(x, CODEBOOK)
        lo, hi = CODEBOOK["edges_m"][path[0]-1:path[0]+1]
        assert abs(decode_scalar(path, CODEBOOK) - x) <= (hi-lo)/64 + 1e-9


def test_later_crop_cannot_leak_into_first_decision():
    # Image at positions 0,1; decision 2; future selected crop at 4,5; decision 6.
    types = [1, 1, 0, 0, 1, 1, 0]
    stages = [0, 0, 0, 1, 1, 1, 1]
    mask = causal_visual_mask(7, types, stages)[0, 0]
    assert mask[0, 1] == 0  # Same initial image sees itself bidirectionally.
    assert mask[4, 5] == 0  # Appended crop also sees itself bidirectionally.
    assert torch.isneginf(mask[:3, 4:6]).all()  # No reverse information path.
    assert mask[6, :].eq(0).all()
    cached = causal_visual_mask(7, types, stages, query_start=3)[0, 0]
    assert torch.equal(cached, mask[3:])


def test_roi_masks_keys_for_new_queries_only_and_preserves_text():
    boxes = np.array([[0, 0, 0, .4, .4], [0, .6, .6, 1, 1]])
    mask = causal_visual_mask(5, [1, 1, 0, 0, 0], [0]*5)
    assert apply_roi_mask(mask, np.array([0, 1]), boxes, (0, 0, .5, .5), 3) == 1
    assert mask[0, 0, 2, 1] == 0
    assert torch.isneginf(mask[0, 0, 3:, 1]).all()
    assert mask[0, 0, 4, 2] == 0
    assert roi_visible(boxes, child_box(ROOT_BOX, 1)).tolist() == [True, False]


def test_pooling_geometry_handles_distinct_global_and_stitched_grids():
    meta = {"token_pooling": np.array([[0, 1, 2, 3], [4, 5, -1, -1], [6, 7, -1, -1]]),
            "subpatch_mapping": [np.array([[4, 5, 6, 7]])]}
    boxes = visual_token_boxes(meta, [[1, 1, 1, 2]], [2], 4)
    assert boxes == pytest.approx(np.array([[0, 0, 0, 1, 1], [0, 0, 0, .5, 1],
                                          [0, .5, 0, 1, 1]]))


def test_one_head_accepts_mixed_candidate_counts_with_masked_gradients():
    head = SharedClassifier(8, 32)
    h = torch.randn(3, 8)
    logits = head(h, torch.tensor([2, 9, 32]))
    assert torch.isneginf(logits[0, 2:]).all()
    assert torch.isneginf(logits[1, 9:]).all()
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 8, 31])).backward()
    assert head.linear.weight.grad[31].abs().sum() > 0


def test_batched_padding_and_per_example_loss_weights():
    examples = []
    for n, positions in [(3, [2]), (5, [2, 4])]:
        examples.append({"inputs": {
            "input_ids": torch.arange(n)[None],
            "attention_mask": causal_visual_mask(n, [0]*n, [0]*n),
            "pixel_values": torch.zeros(1, 4, 3), "image_token_pooling": torch.tensor([[0, 1]]),
            "image_grids": torch.tensor([[1, 1, 0, 0]]), "image_num_crops": torch.tensor([1])},
            "positions": torch.tensor(positions), "counts": torch.tensor([9]*len(positions)),
            "targets": torch.tensor([0]*len(positions))})
    batch = collate_training(examples, 99)
    assert batch["positions"].tolist() == [[0, 2], [1, 2], [1, 4]]
    assert batch["loss_weights"].tolist() == [.5, .25, .25]
    assert batch["inputs"]["input_ids"][0].tolist() == [0, 1, 2, 99, 99]
    assert torch.isneginf(batch["inputs"]["attention_mask"][0, 0, :3, 3:]).all()
    assert batch["inputs"]["image_token_pooling"].tolist() == [[0, 1], [0, 1]]
