import copy
import itertools
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spatial_jev.model import DecisionHeads, last_valid_hidden, task_loss  # noqa: E402
from spatial_jev.schema import (  # noqa: E402
    canonical_metric_question,
    compile_prompt,
    parse_measurement,
    parse_single_point,
)


@pytest.mark.parametrize("answer,expected", [("70 cm", 0.7), ("0.72 m", 0.72),
                                             ("12 inches", 0.3048), ("2 feet", 0.6096)])
def test_physical_unit_normalization(answer, expected):
    assert parse_measurement(answer)[0] == pytest.approx(expected)


def test_measurement_does_not_read_first_number_of_reasoning():
    with pytest.raises(ValueError):
        parse_measurement("Distance[A,B]=1.2m; Distance[A,C]=2.0m. Answer B.")
    with pytest.raises(ValueError):
        canonical_metric_question("Is the object longer than 30 cm?")
    assert "meters" in canonical_metric_question("What is its height in centimeters?")


def test_target_option_survives_every_permutation():
    inp = {"question": "Which object?", "answer_space": {"kind": "choice", "options": [
        {"id": "chair", "text": "chair"}, {"id": "lamp", "text": "lamp"},
        {"id": "table", "text": "table"}]}}
    original = copy.deepcopy(inp)
    for permutation in itertools.permutations(range(3)):
        prompt, mapping = compile_prompt(inp, permutation)
        index = mapping.index("lamp")
        assert f"{index + 1}. lamp" in prompt
        assert "target" not in prompt
    assert original == inp


def test_coordinates_are_xy_and_bounded():
    assert parse_single_point("[(0.2, 0.8)]") == [0.2, 0.8]
    for invalid in ["[(2, 8)]", "[(0.2, 0.8), (0.3, 0.9)]", "[(True, 0.3)]"]:
        with pytest.raises(ValueError):
            parse_single_point(invalid)


def test_pooling_matches_left_and_right_padding():
    h = torch.tensor([[[1.], [2.], [3.]], [[4.], [5.], [6.]]])
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    assert last_valid_hidden(h, mask).flatten().tolist() == [2., 6.]
    with pytest.raises(ValueError):
        last_valid_hidden(h, torch.zeros_like(mask))


def test_choice_loss_ignores_invalid_padding_classes():
    heads = DecisionHeads(8, 5, 4)
    logits = heads(torch.randn(2, 8), "choice", torch.tensor([2, 3]))
    assert torch.isneginf(logits[0, 2:]).all()
    assert torch.isneginf(logits[1, 3:]).all()
    loss = task_loss(logits, torch.tensor([1, 2]), "choice", {})
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.equal(heads.choice.weight.grad[3:], torch.zeros_like(heads.choice.weight.grad[3:]))


def test_scalar_and_point_losses_have_correct_targets():
    config = {"scalar_huber_beta": .1, "point_huber_beta": .05}
    scalar = torch.tensor([math.log1p(.7)], requires_grad=True)
    assert task_loss(scalar, torch.tensor([.7]), "scalar", config).item() == pytest.approx(0)
    point = torch.tensor([[.2, .8]], requires_grad=True)
    assert task_loss(point, torch.tensor([[.2, .8]]), "point", config).item() == 0
