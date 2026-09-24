"""Typed pilot records, strict target parsing and answer-free prompt construction."""

import ast
import math
import re

SCHEMA_VERSION = "spatial-decision/v2"
TASKS = ("choice", "scalar", "point")
UNIT_FACTORS = {"m": 1.0, "meter": 1.0, "meters": 1.0,
                "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
                "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
                "inch": 0.0254, "inches": 0.0254, "ft": 0.3048,
                "foot": 0.3048, "feet": 0.3048}
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def parse_measurement(answer, question=""):
    """Accept a scalar, optionally with an explicit length unit; never extract CoT numbers."""
    match = re.fullmatch(rf"\s*({_NUMBER})\s*([a-zA-Z]+)?\s*\.?\s*", answer)
    if not match:
        raise ValueError("not_a_scalar_answer")
    unit = match.group(2)
    if unit is None:
        units = re.findall(r"\b(" + "|".join(UNIT_FACTORS) + r")\b", question.lower())
        factors = {UNIT_FACTORS[u] for u in units}
        if len(factors) != 1:
            raise ValueError("missing_or_ambiguous_unit")
        unit = units[-1]
    unit = unit.lower()
    if unit not in UNIT_FACTORS:
        raise ValueError("unsupported_unit")
    value = float(match.group(1)) * UNIT_FACTORS[unit]
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid_measurement")
    return value, unit


def parse_single_point(answer):
    try:
        value = ast.literal_eval(answer.strip())
    except (ValueError, SyntaxError) as exc:
        raise ValueError("not_a_point_literal") from exc
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise ValueError("not_a_single_point")
    point = value[0]
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError("not_xy")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in point):
        raise ValueError("non_numeric_point")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in point):
        raise ValueError("point_out_of_range")
    return [float(x) for x in point]


def clean_question(question):
    question = question.replace("<image>", "").strip()
    question = re.split(r"\s*Your answer should be formatted", question, flags=re.I)[0]
    question = re.sub(r"\s*Answer the question using a single word or phrase\.?", "",
                      question, flags=re.I)
    return question.strip()


def canonical_metric_question(question):
    question = clean_question(question)
    # Do not rewrite numeric lengths in a premise without also converting their values.
    if re.search(rf"{_NUMBER}\s*(?:cm|mm|meters?|centimeters?|inches|feet)\b",
                 question, re.I):
        raise ValueError("numeric_unit_in_question_premise")
    return re.sub(r"\b(?:centimeters?|cm|millimeters?|mm|inches|feet|foot|ft)\b",
                  "meters", question, flags=re.I)


def validate_record(record, max_choices=32):
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported_schema")
    inp = record["input"]
    if not inp["question"].strip() or not inp["media"]:
        raise ValueError("empty_input")
    if any(m["kind"] != "image" for m in inp["media"]):
        raise ValueError("pilot_requires_images")
    space, target = inp["answer_space"], record["target"]
    kind = space["kind"]
    if kind == "choice":
        options = space["options"]
        ids = [o["id"] for o in options]
        if not 2 <= len(options) <= max_choices or len(set(ids)) != len(ids):
            raise ValueError("invalid_options")
        if target["option_id"] not in ids:
            raise ValueError("target_missing_from_options")
    elif kind == "scalar":
        if space["unit"] != "m" or target["unit"] != "m":
            raise ValueError("pilot_scalar_requires_meters")
        if not math.isfinite(target["value"]) or target["value"] < 0:
            raise ValueError("invalid_scalar")
    elif kind == "point":
        if len(inp["media"]) != 1:
            raise ValueError("pilot_point_requires_one_image")
        if space["coordinate_system"] != "normalized_xy" or space["num_points"] != 1:
            raise ValueError("unsupported_point_space")
        parse_single_point(repr(target["points"]))
    else:
        raise ValueError("unsupported_task")


def compile_prompt(inp, permutation=None):
    """Only input is accepted; targets and provenance cannot enter the prompt."""
    space = inp["answer_space"]
    kind = space["kind"]
    text = f"Question: {inp['question']}\n"
    option_ids = []
    if kind == "choice":
        options = space["options"]
        order = list(range(len(options))) if permutation is None else list(permutation)
        if sorted(order) != list(range(len(options))):
            raise ValueError("invalid_permutation")
        text += "Options:\n"
        for i, index in enumerate(order):
            option = options[index]
            text += f"{i + 1}. {option['text']}\n"
            option_ids.append(option["id"])
        text += "Select exactly one option."
    elif kind == "scalar":
        text += f"Estimate {space['quantity']}. Return a nonnegative value in meters."
    elif kind == "point":
        text += ("Return one valid point (x, y) in the original image, normalized to [0, 1]. "
                 "The origin is top-left; x increases rightward and y increases downward.")
    else:
        raise ValueError("unsupported_task")
    return text, option_ids
