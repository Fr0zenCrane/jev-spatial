"""Label-isolated scoring for the frozen image-only fast benchmark suite."""

import math
from collections import defaultdict

import numpy as np
from PIL import Image


def point_mask_score(points, mask_path):
    """RefSpatial scorer: normalized xy -> floor pixel index -> binary mask (>0)."""
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask > 0
    if not points:
        return 0.0
    height, width = mask.shape
    values = []
    for point in points:
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(not isinstance(x, (int, float)) or isinstance(x, bool)
                       or not math.isfinite(x) for x in point)):
            values.append(0.0)
            continue
        x, y = point
        if not (0 <= x < 1 and 0 <= y < 1):
            values.append(0.0)
        else:
            values.append(float(mask[int(y * height), int(x * width)]))
    return float(np.mean(values))


def summarize_benchmarks(requests, references, predictions):
    expected = {r["id"] for r in requests}
    if set(references) != expected or set(predictions) != expected:
        raise ValueError("Require one reference and prediction for every request; no dropped failures")
    grouped = defaultdict(list)
    for request in requests:
        key, benchmark = request["id"], request["benchmark"]
        pred, target = predictions[key].get("prediction"), references[key]["target"]
        row = {"id": key, "family": request["family"], "group_id": request["group_id"],
               "valid": pred is not None}
        if benchmark == "sat_real":
            row["score"] = float(pred == target["option_id"])
            row["rotation"] = request["rotation"]
        elif benchmark == "refspatial_bench":
            row["score"] = point_mask_score([pred] if pred is not None else [], target["mask_path"])
        elif benchmark == "vst_numeric_dev":
            valid = isinstance(pred, (int, float)) and math.isfinite(pred) and pred >= 0
            row["valid"] = valid
            row["target"] = target["value"]
            row["error"] = abs(pred - target["value"]) if valid else None
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
        grouped[benchmark].append(row)
    summary = {}
    for benchmark, rows in grouped.items():
        result = {"n_requests": len(rows), "valid_rate": sum(r["valid"] for r in rows) / len(rows)}
        if benchmark in ("sat_real", "refspatial_bench"):
            result["score"] = float(np.mean([r["score"] for r in rows]))
            result["by_family"] = {
                family: {"n": sum(r["family"] == family for r in rows),
                         "score": float(np.mean([r["score"] for r in rows if r["family"] == family]))}
                for family in sorted({r["family"] for r in rows})}
            if benchmark == "sat_real":
                pairs = defaultdict(list)
                for row in rows:
                    pairs[row["group_id"]].append(row["score"])
                if any(len(scores) != 2 for scores in pairs.values()):
                    raise ValueError("SAT protocol requires both orders for every question")
                result["n_questions"] = len(pairs)
                result["both_orders_correct"] = float(np.mean([min(s) for s in pairs.values()]))
                result["metric"] = "accuracy over original+reversed answer orders"
            else:
                result["metric"] = "mean point-in-valid-mask success; singleton output protocol"
        else:
            valid = [r for r in rows if r["valid"]]
            positive = [r for r in rows if r["target"] > 0]
            relative_errors = [r["error"] / r["target"] for r in positive if r["valid"]]
            errors = [r["error"] for r in valid]
            result.update(
                metric="internal in-domain development set; not an official benchmark score",
                mae_m_valid=float(np.mean(errors)) if errors else None,
                mae_m_all=float(np.mean(errors)) if len(valid) == len(rows) else None,
                median_abs_error_m_valid=float(np.median(errors)) if errors else None,
                within_10cm=sum(r["valid"] and r["error"] <= .1 for r in rows) / len(rows),
                within_25pct_positive=sum(r["valid"] and r["error"] / r["target"] <= .25
                                         for r in positive) / len(positive),
                absrel_positive_valid=float(np.mean(relative_errors)) if relative_errors else None)
        summary[benchmark] = result
    return summary
