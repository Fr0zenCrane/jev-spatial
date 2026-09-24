"""Prepare SAT real circular eval + RefSpatial masks + frozen VST development requests.

Inference requests and scoring references are physically separate JSONL files.
"""

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmarks/fast_v1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    images_dir, masks_dir = args.output / "images", args.output / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    requests, references, sources = [], [], {}

    def media(blob):
        sha = digest(blob)
        with Image.open(io.BytesIO(blob)) as image:
            width, height = image.size
            extension = ".png" if image.format == "PNG" else ".jpg" if image.format == "JPEG" else ".img"
        path = images_dir / (sha + extension)
        if not path.exists():
            path.write_bytes(blob)
        return {"kind": "image", "uri": str(path.resolve()), "sha256": sha,
                "width": width, "height": height}

    sat = ROOT / "data/raw/array/SAT/SAT_test.parquet"
    sources[str(sat)] = digest(sat.read_bytes())
    for index, row in enumerate(pq.read_table(sat).to_pylist()):
        if len(row["answers"]) != 2:
            raise ValueError("Pinned SAT original/reverse protocol expects exactly two choices")
        images = [media(blob) for blob in row["image_bytes"]]
        options = [{"id": f"option_{i}", "text": str(answer)} for i, answer in enumerate(row["answers"])]
        matches = [o["id"] for o in options if o["text"] == row["correct_answer"]]
        if len(matches) != 1:
            raise ValueError("Ambiguous SAT correct answer")
        group_id = f"sat_real:{index}"
        for rotation in [0, 1]:
            key = f"{group_id}:order{rotation}"
            ordered = options if rotation == 0 else list(reversed(options))
            requests.append({"id": key, "benchmark": "sat_real", "group_id": group_id,
                             "family": row["question_type"], "rotation": rotation,
                             "input": {"media": images, "question": row["question"],
                                       "answer_space": {"kind": "choice", "options": ordered}}})
            references.append({"id": key, "target": {"option_id": matches[0]}})
    for split in ["location", "placement"]:
        path = ROOT / f"data/raw/BAAI/RefSpatial-Bench/data/{split}-00000-of-00001.parquet"
        sources[str(path)] = digest(path.read_bytes())
        for row in pq.read_table(path).to_pylist():
            key = f"refspatial_bench:{split}:{row['id']}"
            mask_blob = row["mask"]["bytes"]
            mask_path = masks_dir / f"{digest(mask_blob)}.png"
            mask_path.write_bytes(mask_blob)
            # Do not put mask, object description, GT point or reasoning-step annotations in input.
            requests.append({"id": key, "benchmark": "refspatial_bench", "group_id": key,
                             "family": split, "input": {
                                 "media": [media(row["image"]["bytes"])],
                                 "question": row["prompt"], "answer_space": {
                                     "kind": "point", "coordinate_system": "normalized_xy", "num_points": 1}}})
            references.append({"id": key, "target": {"mask_path": str(mask_path.resolve())}})
    dev = ROOT / "data/processed/pilot_v0/20260923T160355Z/dev.jsonl"
    sources[str(dev)] = digest(dev.read_bytes())
    for line in dev.read_text().splitlines():
        row = json.loads(line)
        if row["input"]["answer_space"]["kind"] != "scalar":
            continue
        key = f"vst_numeric_dev:{row['sample_id']}"
        requests.append({"id": key, "benchmark": "vst_numeric_dev", "group_id": row["provenance"]["group_id"],
                         "family": row["provenance"]["family"], "input": row["input"]})
        references.append({"id": key, "target": row["target"]})
    if len(requests) != len({r["id"] for r in requests}):
        raise ValueError("Duplicate benchmark IDs")
    for name, rows in [("requests", requests), ("references", references)]:
        (args.output / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    train = ROOT / "data/processed/pilot_v0/20260923T160355Z/train.jsonl"
    train_hashes = {im["sha256"] for line in train.read_text().splitlines()
                    for im in json.loads(line)["input"]["media"]}
    overlap = {name: sorted({im["sha256"] for r in requests if r["benchmark"] == name
                            for im in r["input"]["media"] if im["sha256"] in train_hashes})
               for name in {r["benchmark"] for r in requests}}
    summary = {"suite": "fast_v1", "n_requests": len(requests),
               "requests_per_benchmark": dict(Counter(r["benchmark"] for r in requests)),
               "source_sha256": sources, "requests_sha256": digest((args.output / "requests.jsonl").read_bytes()),
               "references_sha256": digest((args.output / "references.jsonl").read_bytes()),
               "exact_image_hash_overlap_with_pilot_train": overlap,
               "overlap_limit": "Byte identity only; not a scene audit or base-model training membership audit",
               "protocol": {"sat_real": "150 test questions, original/reverse orders, mean trial accuracy",
                            "refspatial_bench": "100 location + 100 placement, binary mask hit; unseen is overlapping subset",
                            "vst_numeric_dev": "300 existing in-domain held-out development questions, NOT official leaderboard"}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
