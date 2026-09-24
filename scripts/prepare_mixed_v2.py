"""Prepare expanded SAT/VST/real RefSpatial mix with frozen-evaluation media exclusion."""

import argparse
import hashlib
import heapq
import io
import json
import random
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import ijson
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_jev.schema import (SCHEMA_VERSION, canonical_metric_question, clean_question,  # noqa: E402
                                parse_measurement, parse_single_point, validate_record)
from prepare_pilot import balanced_sample, digest  # noqa: E402

PILOT = ROOT / "data/processed/pilot_v0/20260923T160355Z"


def read_jsonl(path):
    return [json.loads(line) for line in path.open()]


def pixel_digest(blob):
    with Image.open(io.BytesIO(blob)) as image:
        rgb = image.convert("RGB")
        return digest(str(rgb.size).encode() + rgb.tobytes()), rgb.size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--wait-until", type=float, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    images = output / "images"
    images.mkdir()
    counters, verified = Counter(), {}
    source_map = {}
    for manifest in ["pilot_v0_files.json", "mixed_v2_assets.json"]:
        for source in json.loads((ROOT / "data/manifests" / manifest).read_text())["sources"]:
            entry = source_map.setdefault(source["repo"], {"revision": source["revision"], "files": {}})
            assert entry["revision"] == source["revision"]
            entry["files"].update({f["path"]: f for f in source["files"]})

    def source_file(repo, name):
        path = ROOT / "data/raw" / repo / name
        expected = source_map[repo]["files"][name]
        while not path.is_file() or path.stat().st_size != expected["bytes"]:
            if time.time() >= args.wait_until:
                raise TimeoutError(f"Source not ready before preparation deadline: {path}")
            print("Waiting for", name, flush=True)
            time.sleep(10)
        if str(path) not in verified:
            with path.open("rb") as stream:
                hasher = hashlib.sha256()
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    hasher.update(chunk)
                sha = hasher.hexdigest()
            if expected.get("sha256") and sha != expected["sha256"]:
                raise ValueError(f"Source hash mismatch: {path}")
            verified[str(path)] = sha
            print("Verified", name, flush=True)
        return path

    fixed_dev = read_jsonl(PILOT / "dev.jsonl")
    fixed_requests = read_jsonl(ROOT / "data/benchmarks/fast_v1/requests.jsonl")
    forbidden_sha, forbidden_rgb, cached = set(), set(), {}
    for row in fixed_dev + fixed_requests:
        for media in row["input"]["media"]:
            sha = media["sha256"]
            forbidden_sha.add(sha)
            if sha not in cached:
                cached[sha] = pixel_digest(Path(media["uri"]).read_bytes())
            forbidden_rgb.add(cached[sha][0])
    print("Frozen eval unique images:", len(forbidden_sha), flush=True)

    def save_image(blob):
        sha = digest(blob)
        if sha in forbidden_sha:
            counters["excluded_eval_image_bytes"] += 1
            return None
        if sha not in cached:
            cached[sha] = pixel_digest(blob)
        rgb, (width, height) = cached[sha]
        if rgb in forbidden_rgb:
            counters["excluded_eval_image_pixels"] += 1
            return None
        path = images / (sha + ".img")
        if not path.exists():
            path.write_bytes(blob)
        return {"kind": "image", "uri": str(path), "sha256": sha,
                "rgb_sha256": rgb, "width": width, "height": height}

    def record(repo, locator, question, media, space, target, family, domain):
        return {"schema_version": SCHEMA_VERSION,
                "sample_id": digest(f"{repo}:{source_map[repo]['revision']}:{locator}"),
                "input": {"media": media, "question": question, "answer_space": space},
                "target": target, "split": "train",
                "provenance": {"source": repo, "revision": source_map[repo]["revision"],
                               "locator": locator, "family": family, "domain": domain}}

    train = []
    repo = "allenai/Molmo2-ER-SAT"
    path = source_file(repo, "SAT_train.parquet")
    rows = pq.read_table(path, columns=["question", "answers", "correct_answer", "question_type"]).to_pylist()
    candidates = []
    for index, row in enumerate(rows):
        options = [str(x).strip().casefold() for x in row["answers"]]
        answer = str(row["correct_answer"]).strip().casefold()
        if 2 <= len(options) <= 32 and len(set(options)) == len(options) and options.count(answer) == 1:
            family = str(row["question_type"])
            if family == "other" and "how many" in row["question"].lower():
                family = "count"
            candidates.append({"index": index, "provenance": {"family": family}})
    selected = {r["index"] for r in balanced_sample(candidates, 35000, args.seed)}
    offset, sat = 0, []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["image_bytes"]):
        for local, packed in enumerate(batch.to_pylist()):
            index = offset + local
            if index not in selected or not 1 <= len(packed["image_bytes"]) <= 4:
                continue
            media = [save_image(blob) for blob in packed["image_bytes"]]
            if any(m is None for m in media):
                continue
            row = rows[index]
            options = [{"id": f"option_{i}", "text": str(x).strip()} for i, x in enumerate(row["answers"])]
            target = next(o["id"] for o in options if o["text"].casefold() == str(row["correct_answer"]).strip().casefold())
            family = str(row["question_type"])
            if family == "other" and "how many" in row["question"].lower():
                family = "count"
            sat.append(record(repo, f"SAT_train.parquet:{index}", clean_question(row["question"]), media,
                              {"kind": "choice", "options": options}, {"option_id": target}, family, "sat"))
        offset += batch.num_rows
    train.extend(balanced_sample(sat, 25000, args.seed))
    del rows, candidates, sat
    print("SAT selected:", len(train), flush=True)

    repo, scalar = "allenai/Molmo2-ER-VST-P", []
    for filename in sorted(source_map[repo]["files"]):
        quantity = "height" if "object_height" in filename else "longest_dimension"
        domain = "hypersim" if "hypersim" in filename else "real"
        offset = 0
        for batch in pq.ParquetFile(source_file(repo, filename)).iter_batches(batch_size=8):
            for local, row in enumerate(batch.to_pylist()):
                qa, turns = [], row["conversations"]
                if not 1 <= len(row["images"]) <= 4:
                    continue
                for i in range(0, len(turns) - 1, 2):
                    if turns[i]["from"] != "human" or turns[i + 1]["from"] != "gpt":
                        continue
                    try:
                        question = canonical_metric_question(turns[i]["value"])
                        value, unit = parse_measurement(turns[i + 1]["value"], turns[i]["value"])
                    except ValueError as exc:
                        counters[f"vst_{exc}"] += 1
                        continue
                    qa.append((i, question, value, unit))
                # Limit repeated views without choosing examples based on scalar magnitude.
                random.Random(args.seed + offset + local).shuffle(qa)
                qa = qa[:6]
                if not qa:
                    continue
                media = [save_image(im["bytes"]) for im in row["images"]]
                if any(m is None for m in media):
                    continue
                for i, question, value, unit in qa:
                    item = record(repo, f"{filename}:{offset + local}:{i}", question, media,
                                  {"kind": "scalar", "quantity": quantity, "unit": "m"},
                                  {"value": value, "unit": "m"}, quantity, domain)
                    item["provenance"]["original_unit"] = unit
                    scalar.append(item)
            offset += batch.num_rows
        print("VST scanned:", filename, "rows", offset, "candidates", len(scalar), flush=True)
    real = balanced_sample([r for r in scalar if r["provenance"]["domain"] == "real"], 20000, args.seed)
    sim = balanced_sample([r for r in scalar if r["provenance"]["domain"] == "hypersim"], 5000, args.seed)
    train.extend(real + sim)
    del scalar, real, sim

    repo = "allenai/Molmo2-ER-RefSpatial"
    heaps = defaultdict(list)
    for filename, family in [("3D/reasoning_template_qa.json", "object"), ("3D/vacant_qa.json", "placement")]:
        with source_file(repo, filename).open("rb") as stream:
            for index, row in enumerate(ijson.items(stream, "item", use_float=True)):
                if len(row.get("image", [])) != 1:
                    continue
                name = Path(row["image"][0]).name
                # Conservative filename-prefix holdout; physical scene identity is not certified.
                group = name.split("_")[0]
                split = "dev" if int(digest(f"{args.seed}:ref3d:{group}")[:8], 16) % 10 == 0 else "train"
                turns, candidates = row.get("conversations", []), []
                for i in range(0, len(turns) - 1, 2):
                    if turns[i]["from"] != "human" or turns[i + 1]["from"] != "gpt":
                        continue
                    try:
                        xy = parse_single_point(turns[i + 1]["value"])
                    except ValueError:
                        continue
                    candidates.append((i, xy))
                counters[f"ref_{family}_valid_qa"] += len(candidates)
                random.Random(args.seed + index).shuffle(candidates)
                for i, xy in candidates[:2]:
                    item = record(repo, f"{filename}:{index}:{i}", clean_question(turns[i]["value"]),
                                  [{"kind": "image", "source_name": name}],
                                  {"kind": "point", "coordinate_system": "normalized_xy", "num_points": 1},
                                  {"points": [xy]}, family, "ref3d")
                    item["split"] = split
                    item["provenance"].update(filename_group=group, source_image=name)
                    priority = int(digest(f"{args.seed}:{item['sample_id']}")[:16], 16)
                    entry = (-priority, item["sample_id"], item)
                    heap = heaps[(split, family)]
                    limit = 13000 if split == "train" else 400
                    if len(heap) < limit:
                        heapq.heappush(heap, entry)
                    elif entry > heap[0]:
                        heapq.heapreplace(heap, entry)
                if index and index % 50000 == 0:
                    print("Ref scanned", filename, index, flush=True)
        print("Ref pools", {str(k): len(v) for k, v in heaps.items()}, flush=True)
    points = [entry[2] for heap in heaps.values() for entry in heap]
    needed = {r["input"]["media"][0]["source_name"] for r in points}
    found, names_seen = {}, set()
    with tarfile.open(source_file(repo, "3D/image/image.tar.gz"), "r|gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if member.isfile() and name in needed:
                if name in names_seen:
                    raise ValueError(f"Ambiguous basename: {name}")
                names_seen.add(name)
                found[name] = save_image(archive.extractfile(member).read())
    if needed - found.keys():
        raise ValueError(f"Missing Ref images: {len(needed - found.keys())}")
    resolved = []
    for item in points:
        media = found[item["input"]["media"][0]["source_name"]]
        if media is not None:
            item["input"]["media"] = [media]
            resolved.append(item)
    train.extend(balanced_sample([r for r in resolved if r["split"] == "train"], 20000, args.seed))

    old_train = read_jsonl(PILOT / "train.jsonl")
    replay = [r for r in old_train if r["input"]["answer_space"]["kind"] == "point"]
    # Old training images must also not enter the new supplementary dev split.
    exposed_rgb = set()
    for row in old_train:
        for media in row["input"]["media"]:
            sha = media["sha256"]
            if sha not in cached:
                cached[sha] = pixel_digest(Path(media["uri"]).read_bytes())
            exposed_rgb.add(cached[sha][0])
    for row in replay:
        for media in row["input"]["media"]:
            media["rgb_sha256"] = cached[media["sha256"]][0]
        row["provenance"]["domain"] = "simulator_replay"
    train.extend(replay)
    new_dev = balanced_sample([r for r in resolved if r["split"] == "dev"
                               and r["input"]["media"][0]["rgb_sha256"] not in exposed_rgb], 400, args.seed)
    dev_rgb = {m["rgb_sha256"] for r in new_dev for m in r["input"]["media"]}
    train = [r for r in train if not any(m["rgb_sha256"] in dev_rgb for m in r["input"]["media"])]
    unique, conflicts = {}, set()
    for row in train:
        validate_record(row)
        inp = row["input"]
        key = digest(json.dumps([inp["question"], inp["answer_space"],
                                 [m["rgb_sha256"] for m in inp["media"]]], sort_keys=True))
        if key in unique:
            counters["duplicate_inputs"] += 1
            if row["target"] != unique[key]["target"]:
                conflicts.add(key)
        else:
            unique[key] = row
    train = [r for key, r in unique.items() if key not in conflicts]
    counters["conflicting_inputs"] = len(conflicts)
    assert not forbidden_rgb & {m["rgb_sha256"] for r in train for m in r["input"]["media"]}
    if len(new_dev) < 100:
        raise ValueError("Insufficient real Ref supplementary dev")
    # Real pointing dev comes first for the trainer's small, fixed validation subset.
    dev = new_dev + fixed_dev
    stats = {}
    for split, rows in [("train", train), ("dev", dev), ("real_point_dev", new_dev)]:
        if split == "train":
            random.Random(args.seed).shuffle(rows)
        with (output / f"{split}.jsonl").open("w") as stream:
            for row in rows:
                validate_record(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[split] = {"tasks": dict(Counter(r["input"]["answer_space"]["kind"] for r in rows)),
                        "domains": dict(Counter(r["provenance"].get("domain", "frozen_pilot_dev") for r in rows)),
                        "task_families": dict(Counter(r["input"]["answer_space"]["kind"] + "/" + r["provenance"]["family"] for r in rows)),
                        "unique_images": len({m["sha256"] for r in rows for m in r["input"]["media"]}),
                        "sha256": digest((output / f"{split}.jsonl").read_bytes())}
    values = [r["target"]["value"] for r in train if r["input"]["answer_space"]["kind"] == "scalar"]
    summary = {"seed": args.seed, "splits": stats, "filters": dict(counters),
               "verified_source_sha256": verified,
               "scalar_m_quantiles": dict(zip(["min", "p50", "p90", "p95", "p99", "max"],
                                               np.quantile(values, [0, .5, .9, .95, .99, 1]).tolist())),
               "scalar_over_20m": sum(v > 20 for v in values),
               "scalar_policy": "strict parse, canonical meters, flag only; unchanged codebook",
               "split_guarantee": "byte and decoded RGB exclusion of fixed dev and fast suite; new Ref dev holds out filename prefixes; no certified full scene or base-model-membership audit",
               "benchmark_labels_used": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
