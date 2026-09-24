"""Answer-only generation prompts and label-independent parsers for pilot evaluation."""

import ast
import math
import random
import re

from .schema import compile_prompt, parse_measurement

REFSPATIAL_POINT_SUFFIX = (
    " Your answer should be formatted as a list of tuples, i.e. [(x1, y1)], "
    "where each tuple contains the x and y coordinates of a point satisfying the "
    "conditions above. The coordinates should be between 0 and 1, indicating the "
    "normalized pixel locations of the points in the image."
)


def generation_prompt(inp, seed, point_source_prompt=False):
    kind = inp["answer_space"]["kind"]
    if point_source_prompt:
        if kind != "point":
            raise ValueError("source prompt profile only supports RefSpatial pointing")
        return inp["question"] + REFSPATIAL_POINT_SUFFIX, []
    order = None
    if kind == "choice":
        order = list(range(len(inp["answer_space"]["options"])))
        random.Random(seed).shuffle(order)
    prompt, mapping = compile_prompt(inp, order)
    suffix = {
        "choice": f"Answer with only the option number (1 to {len(mapping)}).",
        "scalar": "Answer with only the number and unit.",
        "point": "Answer with only one (x, y) pair.",
    }[kind]
    return prompt + "\n" + suffix, mapping


def parse_generated_answer(text, inp, mapping):
    """Never inspect targets or infer coordinate scales from the reference answer."""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)[:-3].strip()
    text = re.sub(r"^(?:answer|final answer)\s*:\s*", "", text, flags=re.I)
    task = inp["answer_space"]["kind"]
    if task == "choice":
        match = re.fullmatch(r"(?:option\s*)?\(?([0-9]+)\)?[.]?", text, re.I)
        if match:
            index = int(match.group(1)) - 1
            if 0 <= index < len(mapping):
                return mapping[index], "option_number"
            raise ValueError("option_number_out_of_range")
        options = inp["answer_space"]["options"]
        matches = [o["id"] for o in options
                   if o["text"].strip().rstrip(".").casefold() == text.rstrip(".").casefold()]
        if len(matches) == 1:
            return matches[0], "exact_option_text"
        raise ValueError("not_an_unambiguous_choice")
    if task == "scalar":
        value, _ = parse_measurement(text, "Return a value in meters.")
        return value, "scalar_with_unit_or_requested_meters"
    if task != "point":
        raise ValueError("unsupported_task")
    # Molmo2 unified HTML format: image_id point_id x_1000 y_1000.
    match = re.fullmatch(r'<points(?:\s+alt="[^"]*")?\s+coords="([\d ]+)">'
                         r'[^<>]*</points>', text)
    if match:
        parts = match.group(1).split()
        if len(parts) != 4 or parts[:2] != ["1", "1"]:
            raise ValueError("not_a_single_image_single_point")
        value = [int(parts[2]) / 1000, int(parts[3]) / 1000]
        mode = "molmo2_html_1000"
    else:
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("not_a_coordinate_answer") from exc
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        mode = "normalized_xy_literal"
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("not_exactly_one_xy_pair")
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) or not 0 <= x <= 1 for x in value):
        raise ValueError("invalid_normalized_xy")
    return list(map(float, value)), mode


def score_prediction(row, prediction):
    task = row["input"]["answer_space"]["kind"]
    target = row["target"]
    if task == "choice":
        return {"correct": prediction == target["option_id"]}
    if prediction is None:
        return {}
    if task == "scalar":
        error = abs(prediction - target["value"])
        return {"abs_error_m": error, "relative_error": (
            error / target["value"] if target["value"] > 0 else None)}
    reference = target["points"][0]
    dx, dy = prediction[0] - reference[0], prediction[1] - reference[1]
    media = row["input"]["media"][0]
    return {"normalized_l2": math.hypot(dx, dy),
            "pixel_l2": math.hypot(dx * media["width"], dy * media["height"])}
