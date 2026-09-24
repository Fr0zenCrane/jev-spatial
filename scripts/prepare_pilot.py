"""Build a reproducible, media-grouped three-task pilot from the selected raw files."""

import argparse
import hashlib
import heapq
import io
import json
import random
import sys
import tarfile
import time
import subprocess
from collections import Counter, defaultdict, deque
from pathlib import Path

import ijson
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spatial_jev.schema import (  # noqa: E402
    SCHEMA_VERSION,
    canonical_metric_question,
    clean_question,
    parse_measurement,
    parse_single_point,
    validate_record,
)


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def grouped_split(group, seed):
    return "dev" if int(digest(f"{seed}:{group}")[:8], 16) % 10 == 0 else "train"


def balanced_sample(records, count, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for record in records:
        buckets[record["provenance"]["family"]].append(record)
    queues = []
    for key in sorted(buckets):
        rng.shuffle(buckets[key])
        queues.append(deque(buckets[key]))
    selected = []
    while queues and len(selected) < count:
        remaining = []
        for queue in queues:
            if len(selected) < count:
                selected.append(queue.popleft())
            if queue:
                remaining.append(queue)
        queues = remaining
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--wait-for-inputs", action="store_true")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new preparation output directory")
    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "images"
    images_dir.mkdir()
    manifest = json.loads((root / "data/manifests/pilot_v0_files.json").read_text())
    sources = {s["repo"]: s for s in manifest["sources"]}
    records = []
    counters = Counter()

    def source_file(repo, filename):
        path = root / "data/raw" / repo / filename
        expected = next(f["bytes"] for f in sources[repo]["files"] if f["path"] == filename)
        while not path.exists() or path.stat().st_size != expected:
            if not args.wait_for_inputs:
                raise ValueError(f"Missing or incomplete source: {path}")
            alive = subprocess.run(["tmux", "has-session", "-t", "optimus-pilot-ranges"],
                                   capture_output=True)
            if alive.returncode:
                raise ValueError(f"Downloader stopped before source became complete: {path}")
            print("Waiting for verified source:", path, flush=True)
            time.sleep(10)
        return path

    def save_image(blob):
        key = digest(blob)
        with Image.open(io.BytesIO(blob)) as image:
            width, height = image.size
            extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image.format)
            if extension is None:
                raise ValueError("unsupported_image_format")
            image.verify()
        path = images_dir / (key + extension)
        if not path.exists():
            path.write_bytes(blob)
        return {"kind": "image", "uri": str(path), "sha256": key,
                "width": width, "height": height}

    def record(repo, locator, question, media, space, target, family):
        return {"schema_version": SCHEMA_VERSION,
                "sample_id": digest(f"{repo}:{sources[repo]['revision']}:{locator}"),
                "input": {"media": media, "question": question, "answer_space": space},
                "target": target,
                "provenance": {"source": repo, "revision": sources[repo]["revision"],
                               "locator": locator, "family": family}}

    # SAT: sample text records before reading the large embedded-image column.
    repo = "allenai/Molmo2-ER-SAT"
    path = source_file(repo, "SAT_train.parquet")
    table = pq.read_table(path, columns=["question", "answers", "correct_answer", "question_type"])
    rows = table.to_pylist()
    valid = []
    for index, row in enumerate(rows):
        options = [str(x).strip() for x in row["answers"]]
        normalized = [x.casefold() for x in options]
        answer = str(row["correct_answer"]).strip().casefold()
        if not 2 <= len(options) <= 32 or len(set(normalized)) != len(options):
            counters["sat_invalid_options"] += 1
            continue
        if normalized.count(answer) != 1:
            counters["sat_unmapped_answer"] += 1
            continue
        valid.append(index)
    random.Random(args.seed).shuffle(valid)
    selected = set(valid[:14000])
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["image_bytes"]):
        for local, packed in enumerate(batch.to_pylist()):
            index = offset + local
            if index not in selected:
                continue
            blobs = packed["image_bytes"]
            if not 1 <= len(blobs) <= 4:
                counters["sat_image_budget"] += 1
                continue
            row = rows[index]
            media = [save_image(blob) for blob in blobs]
            options = [{"id": f"option_{i}", "text": str(x).strip()}
                       for i, x in enumerate(row["answers"])]
            target = next(o["id"] for o in options if o["text"].casefold()
                          == str(row["correct_answer"]).strip().casefold())
            family = str(row["question_type"])
            if family == "other" and "how many" in row["question"].lower():
                family = "count"
            records.append(record(repo, f"SAT_train.parquet:{index}",
                                  clean_question(row["question"]), media,
                                  {"kind": "choice", "options": options},
                                  {"option_id": target}, family))
        offset += batch.num_rows
    del rows, table
    print("SAT candidate QA:", len(records), flush=True)

    # VST: strict scalar answers; multi-turn rows are expanded without assistant history.
    repo = "allenai/Molmo2-ER-VST-P"
    for file in sources[repo]["files"]:
        quantity = "height" if "object_height" in file["path"] else "longest_dimension"
        offset = 0
        for batch in pq.ParquetFile(source_file(repo, file["path"])).iter_batches(batch_size=8):
            for local, row in enumerate(batch.to_pylist()):
                qa = []
                turns = row["conversations"]
                for i in range(0, len(turns) - 1, 2):
                    if turns[i]["from"] != "human" or turns[i + 1]["from"] != "gpt":
                        counters["vst_invalid_roles"] += 1
                        continue
                    try:
                        question = canonical_metric_question(turns[i]["value"])
                        value, original_unit = parse_measurement(
                            turns[i + 1]["value"], turns[i]["value"])
                    except ValueError as exc:
                        counters[f"vst_{exc}"] += 1
                        continue
                    qa.append((i, question, value, original_unit))
                if not qa or not 1 <= len(row["images"]) <= 4:
                    continue
                media = [save_image(im["bytes"]) for im in row["images"]]
                for i, question, value, original_unit in qa:
                    item = record(repo, f"{file['path']}:{offset + local}:{i}", question,
                                  media, {"kind": "scalar", "quantity": quantity, "unit": "m"},
                                  {"value": value, "unit": "m"}, quantity)
                    item["provenance"]["original_unit"] = original_unit
                    records.append(item)
            offset += batch.num_rows
    print("QA after VST:", len(records), flush=True)

    # RefSpatial: deterministic reservoir across the complete annotation stream.
    repo = "allenai/Molmo2-ER-RefSpatial"
    heaps = {"object": [], "placement": []}
    with source_file(repo, "Simulator/metadata.json").open("rb") as stream:
        for index, row in enumerate(ijson.items(stream, "item", use_float=True)):
            if len(row.get("image", [])) != 1:
                counters["ref_multi_image"] += 1
                continue
            name = Path(row["image"][0]).name
            turns = row.get("conversations", [])
            for i in range(0, len(turns) - 1, 2):
                if turns[i]["from"] != "human" or turns[i + 1]["from"] != "gpt":
                    continue
                try:
                    xy = parse_single_point(turns[i + 1]["value"])
                except ValueError as exc:
                    counters[f"ref_{exc}"] += 1
                    continue
                question = clean_question(turns[i]["value"])
                family = "placement" if any(t in question.lower() for t in
                                            ["vacant", "free space", "free location", "free point",
                                             "free spot", "unoccupied"]) else "object"
                item = record(repo, f"Simulator/metadata.json:{index}:{i}", question,
                              [{"kind": "image", "source_name": name}],
                              {"kind": "point", "coordinate_system": "normalized_xy",
                               "num_points": 1}, {"points": [xy]}, family)
                priority = int(digest(f"{args.seed}:{item['sample_id']}")[:16], 16)
                heap = heaps[family]
                entry = (-priority, item["sample_id"], item)
                if len(heap) < 16000:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)
            if index and index % 20000 == 0:
                print("Ref annotation records scanned:", index, flush=True)
    points = [entry[2] for heap in heaps.values() for entry in heap]
    needed = {r["input"]["media"][0]["source_name"] for r in points}
    found = {}
    with tarfile.open(source_file(repo, "Simulator/image/image.tar.gz"), "r|gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if member.isfile() and name in needed:
                if name in found:
                    raise ValueError(f"Ambiguous image basename in archive: {name}")
                found[name] = save_image(archive.extractfile(member).read())
    missing = needed - found.keys()
    if missing:
        raise ValueError(f"Selected annotations reference {len(missing)} missing images")
    for item in points:
        item["input"]["media"] = [found[item["input"]["media"][0]["source_name"]]]
    records.extend(points)
    print("QA candidates:", len(records), flush=True)

    # Remove exact duplicate inputs, and discard inputs with conflicting labels.
    unique, conflicts = {}, set()
    for item in records:
        validate_record(item)
        key = digest(json.dumps(item["input"], sort_keys=True))
        if key in unique:
            counters["duplicate_input"] += 1
            if item["target"] != unique[key]["target"]:
                conflicts.add(key)
        else:
            unique[key] = item
    records = [item for key, item in unique.items() if key not in conflicts]
    counters["conflicting_inputs_removed"] = len(conflicts)

    # Connected components prevent an image reused in different multi-image requests leaking.
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    for item in records:
        keys = [m["sha256"] for m in item["input"]["media"]]
        for key in keys[1:]:
            a, b = find(keys[0]), find(key)
            parents[max(a, b)] = min(a, b)
    pools = defaultdict(list)
    for item in records:
        group = find(item["input"]["media"][0]["sha256"])
        item["provenance"]["group_id"] = group
        item["provenance"]["split_guarantee"] = "exact_media_connected_component"
        item["split"] = grouped_split(group, args.seed)
        pools[(item["split"], item["input"]["answer_space"]["kind"])].append(item)
    counts = {"train": {"choice": 5000, "scalar": 3000, "point": 5000},
              "dev": {"choice": 500, "scalar": 300, "point": 500}}
    summaries, selected_records = {}, {}
    for split, limits in counts.items():
        chosen = []
        for task, limit in limits.items():
            subset = balanced_sample(pools[(split, task)], limit, args.seed)
            if len(subset) < min(limit, 100):
                raise ValueError(f"Insufficient {split}/{task} examples: {len(subset)}")
            chosen.extend(subset)
        random.Random(args.seed).shuffle(chosen)
        with (output / f"{split}.jsonl").open("w") as stream:
            for item in chosen:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        selected_records[split] = chosen
        summaries[split] = {"tasks": dict(Counter(r["input"]["answer_space"]["kind"]
                                                 for r in chosen)),
                            "groups": len({r["provenance"]["group_id"] for r in chosen}),
                            "jsonl_sha256": digest((output / f"{split}.jsonl").read_bytes())}
    groups = [{r["provenance"]["group_id"] for r in selected_records[s]} for s in counts]
    assert not groups[0] & groups[1]
    summary = {"seed": args.seed, "source_manifest_sha256": digest(
        (root / "data/manifests/pilot_v0_files.json").read_bytes()),
        "splits": summaries, "filter_counts": dict(counters),
        "split_scope": "Exact-media grouping only; original physical scene IDs unavailable.",
        "benchmark_used": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
