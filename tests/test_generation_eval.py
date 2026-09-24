import random

import pytest

from spatial_jev.generation_eval import generation_prompt, parse_generated_answer, score_prediction
from spatial_jev.schema import compile_prompt


def test_generation_preserves_eval_option_permutation():
    inp = {"question": "How many?", "answer_space": {"kind": "choice", "options": [
        {"id": "a", "text": "3"}, {"id": "b", "text": "1"}, {"id": "c", "text": "2"}]}}
    order = list(range(3))
    random.Random(987).shuffle(order)
    expected, ids = compile_prompt(inp, order)
    prompt, mapping = generation_prompt(inp, 987)
    assert mapping == ids
    assert prompt.startswith(expected + "\n")
    # A numeric response denotes displayed option position, never target count.
    assert parse_generated_answer("2", inp, mapping)[0] == mapping[1]


@pytest.mark.parametrize("text,expected", [("70 cm", .7), ("0.7", .7), ("2 feet", .6096)])
def test_generated_scalar_units(text, expected):
    inp = {"answer_space": {"kind": "scalar"}}
    assert parse_generated_answer(text, inp, [])[0] == pytest.approx(expected)


@pytest.mark.parametrize("text", ["It may be 1 or 2 meters", "-2 m", "nan", "1 to 2 m"])
def test_ambiguous_scalar_is_not_scored_as_first_number(text):
    with pytest.raises(ValueError):
        parse_generated_answer(text, {"answer_space": {"kind": "scalar"}}, [])


def test_point_formats_and_no_implicit_rescaling():
    inp = {"answer_space": {"kind": "point"}}
    for text in ["(0.72, 0.55)", "[(0.72, 0.55)]",
                 '<points coords="1 1 720 550">cup</points>']:
        assert parse_generated_answer(text, inp, [])[0] == [.72, .55]
    for text in ["(720, 550)", "[(.1, .2), (.3, .4)]", "(True, .5)",
                 '<points coords="1 1 720 550 2 200 100">cups</points>']:
        with pytest.raises(ValueError):
            parse_generated_answer(text, inp, [])


def test_missing_answer_is_a_failure_and_point_pixels_use_original_size():
    row = {"input": {"answer_space": {"kind": "choice"}}, "target": {"option_id": "a"}}
    assert score_prediction(row, None) == {"correct": False}
    row = {"input": {"answer_space": {"kind": "point"},
                     "media": [{"width": 1000, "height": 500}]},
           "target": {"points": [[.3, .4]]}}
    assert score_prediction(row, [.4, .6])["pixel_l2"] == pytest.approx(2 ** .5 * 100)


def test_source_point_prompt_retains_question_and_declares_normalization():
    inp = {"question": "Identify the basket.", "answer_space": {"kind": "point"}}
    prompt, mapping = generation_prompt(inp, 0, point_source_prompt=True)
    assert prompt.startswith(inp["question"] + " Your answer should be formatted")
    assert "[(x1, y1)]" in prompt and "between 0 and 1" in prompt
    assert mapping == []
