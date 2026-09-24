import torch
import torch.nn.functional as F

from spatial_jev.scene_inference import branch_mask
from spatial_jev.runtime import causal_visual_mask


def test_question_branches_cannot_read_each_other_or_their_own_future():
    mask = branch_mask(2, [-1, -1, 0, 0, 1, 0], [0, 1, 2, 3, 2, 4], 2, 'cpu')
    expected = torch.tensor([[1, 1, 1, 0, 0, 0],
                             [1, 1, 1, 1, 0, 0],
                             [1, 1, 0, 0, 1, 0],
                             [1, 1, 1, 1, 0, 1]], dtype=torch.bool)
    assert torch.equal(mask[0, 0] == 0, expected)


def test_packed_attention_matches_separate_question_attention():
    torch.manual_seed(7)
    query, key, value = [torch.randn(1, 2, 6, 8) for _ in range(3)]
    mask = branch_mask(2, [-1, -1, 0, 0, 1, 0], [0, 1, 2, 3, 2, 4], 2, 'cpu')
    packed = F.scaled_dot_product_attention(query[:, :, 2:], key, value, attn_mask=mask)
    visible_keys = [[0, 1, 2], [0, 1, 2, 3], [0, 1, 4], [0, 1, 2, 3, 5]]
    for i, indices in enumerate(visible_keys):
        independent = F.scaled_dot_product_attention(
            query[:, :, i+2:i+3], key[:, :, indices], value[:, :, indices])
        torch.testing.assert_close(packed[:, :, i:i+1], independent)


def test_crop_attention_matches_each_independent_reveal_sequence():
    # Two queries, each revealing its own two-token crop after a textual question.
    owners = [-1, -1, 0, 1, 0, 0, 0, 1, 1, 1]
    positions = [0, 1, 2, 2, 3, 4, 5, 3, 4, 5]
    visual = [0, 1, 0, 0, 1, 1, 0, 1, 1, 0]
    stages = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    packed = branch_mask(2, owners, positions, 4, 'cpu', visual, stages)
    for owner, indices, query_rows in [(0, [0, 1, 2, 4, 5, 6], [0, 1, 2]),
                                      (1, [0, 1, 3, 7, 8, 9], [3, 4, 5])]:
        expected = causal_visual_mask(6, [visual[i] for i in indices],
                                      [stages[i] for i in indices], query_start=3)
        assert torch.equal(packed[:, :, query_rows][:, :, :, indices], expected)
        other = [i for i, o in enumerate(owners) if o not in (-1, owner)]
        assert torch.isneginf(packed[:, :, query_rows][:, :, :, other]).all()
