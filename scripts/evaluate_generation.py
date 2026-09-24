"""Evaluate unadapted Molmo2-ER on the identical pilot development examples.

Run with torchrun for independent per-GPU shards, then use --summarize to merge.
Raw answers and prompts are saved; invalid/truncated answers are never dropped silently.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_jev.generation_eval import (  # noqa: E402
    generation_prompt,
    parse_generated_answer,
    score_prediction,
)


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def metrics(rows, task):
    import numpy as np

    valid = [r for r in rows if r["parse_error"] is None]
    n = len(rows)
    result = {"n": n, "valid_n": len(valid), "parse_success_rate": len(valid) / n,
              "parse_errors": dict(Counter(r["parse_error"] for r in rows if r["parse_error"])),
              "truncated_n": sum(r["truncated"] for r in rows)}
    if task == "choice":
        result["accuracy_all"] = sum(r["scores"]["correct"] for r in rows) / n
        return result
    if task == "scalar":
        errors = [r["scores"]["abs_error_m"] for r in valid]
        rel = [r["scores"]["relative_error"] for r in valid
               if r["scores"]["relative_error"] is not None]
        positive_n = sum(r["target"]["value"] > 0 for r in rows)
        result.update(mae_m_valid=float(np.mean(errors)) if errors else None,
                      median_abs_error_m_valid=float(np.median(errors)) if errors else None,
                      p90_abs_error_m_valid=float(np.quantile(errors, .9)) if errors else None,
                      abs_rel_positive_valid=float(np.mean(rel)) if rel else None,
                      positive_target_n=positive_n,
                      within_10cm_all=sum(x <= .1 for x in errors) / n,
                      within_25pct_positive_all=(sum(x <= .25 for x in rel) / positive_n
                                                 if positive_n else None))
        # No arbitrary numerical penalty for parse failures: all-sample MAE is undefined.
        result["mae_m_all"] = result["mae_m_valid"] if len(valid) == n else None
    else:
        errors = [r["scores"]["normalized_l2"] for r in valid]
        result.update(mean_l2_valid=float(np.mean(errors)) if errors else None,
                      median_l2_valid=float(np.median(errors)) if errors else None,
                      p90_l2_valid=float(np.quantile(errors, .9)) if errors else None,
                      mean_pixel_l2_valid=(float(np.mean([r["scores"]["pixel_l2"] for r in valid]))
                                           if valid else None),
                      hit_at_005_all=sum(x <= .05 for x in errors) / n,
                      hit_at_010_all=sum(x <= .1 for x in errors) / n)
        result["mean_l2_all"] = result["mean_l2_valid"] if len(valid) == n else None
    return result


def summarize(args, selected, config):
    records = []
    for path in sorted(args.output.glob("predictions-rank*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    by_id = {r["sample_id"]: r for r in records}
    expected = [r["sample_id"] for r in selected]
    if len(by_id) != len(records) or set(by_id) != set(expected):
        raise ValueError("Missing, duplicate, or unexpected evaluation records")
    ordered = [by_id[sample_id] for sample_id in expected]
    with (args.output / "predictions.jsonl").open("w") as stream:
        for row in ordered:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    result = {"config": config, "completed_at": datetime.now(timezone.utc).isoformat(),
              "metrics": {}}
    for task in ("choice", "scalar", "point"):
        rows = [r for r in ordered if r["task"] == task]
        if not rows:
            continue
        result["metrics"][task] = {
            "overall": metrics(rows, task), "by_family": {
                family: metrics([r for r in rows if r["family"] == family], task)
                for family in sorted({r["family"] for r in rows})}}
        if task == "scalar":
            result["metrics"][task]["by_size"] = {
                name: metrics(subset, task) for name, subset in (
                    ("lt_1m", [r for r in rows if r["target"]["value"] < 1]),
                    ("1_to_3m", [r for r in rows if 1 <= r["target"]["value"] < 3]),
                    ("ge_3m", [r for r in rows if r["target"]["value"] >= 3])) if subset}
    write_json(args.output / "summary.json", result)
    print(json.dumps(result["metrics"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot_v0.json"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--point-source-prompt", action="store_true",
                        help="Evaluate only pointing, using its original RefSpatial prompt template")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    records = [json.loads(line) for line in (args.data / "dev.jsonl").read_text().splitlines()]
    pools = [[r for r in records if r["input"]["answer_space"]["kind"] == task]
             for task in ("choice", "scalar", "point")]
    if args.point_source_prompt:
        pools = [pools[2]]
    if args.limit_per_task:
        pools = [rows[:args.limit_per_task] for rows in pools]
    # Interleave tasks so each independent worker has a representative shard.
    selected = [pool[i] for i in range(max(map(len, pools))) for pool in pools if i < len(pool)]
    run_config = {**config, "baseline": "unadapted_native_generation", "prompt_version": 1,
                  "prompt_policy": "pilot compile_prompt plus answer-only format suffix",
                  "choice_option_policy": "same seed XOR sample_id fixed shuffle as head eval",
                  "decoding": {"do_sample": False, "num_beams": 1,
                               "max_new_tokens": args.max_new_tokens, "use_cache": True},
                  "limit_per_task": args.limit_per_task,
                  "dev_sha256": hashlib.sha256((args.data / "dev.jsonl").read_bytes()).hexdigest(),
                  "selected_n": len(selected),
                  "parse_policy": "strict, no ground-truth-based extraction or scale inference",
                  "invalid_policy": "failure on all-sample success metrics; MAE valid-only labeled",
                  "benchmark_status": "internal dev; base-model training overlap possible"}
    if args.point_source_prompt:
        run_config["prompt_policy"] = "original RefSpatial question and normalized list-of-tuples suffix"
        run_config["point_source_prompt"] = True
    if args.summarize:
        if json.loads((args.output / "config.json").read_text()) != run_config:
            raise ValueError("Summary configuration differs from generation configuration")
        summarize(args, selected, run_config)
        return
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText

    from spatial_jev.batching import make_processor

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        write_json(args.output / "config.json", run_config)
        write_json(args.output / "environment.json", {
            "started_at": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device), "workers": world})
        for path in [Path(__file__), ROOT / "src/spatial_jev/generation_eval.py"]:
            shutil.copy2(path, args.output / path.name)
    started = time.monotonic()
    base = str((ROOT / config["base_model"]).resolve())
    processor = make_processor(base, config["max_crops"])
    model = AutoModelForImageTextToText.from_pretrained(
        base, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").to(device).eval()
    shard = selected[rank::world]
    output = args.output / f"predictions-rank{rank:02d}.jsonl"
    with output.open("x") as stream:
        for index, row in enumerate(shard):
            inp = row["input"]
            task = inp["answer_space"]["kind"]
            prompt, mapping = generation_prompt(
                inp, config["seed"] ^ int(row["sample_id"][:8], 16), args.point_source_prompt)
            content = [{"type": "image"} for _ in inp["media"]]
            content.append({"type": "text", "text": prompt})
            text = processor.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=True)
            images = []
            for media in inp["media"]:
                with Image.open(media["uri"]) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
            length = inputs["input_ids"].shape[1]
            if length > config["max_sequence_length"] or len(images) > config["max_images"]:
                raise ValueError("Input exceeds matched pilot media/token budget")
            inputs = {key: value.to(device=device, dtype=torch.bfloat16)
                      if value.is_floating_point() else value.to(device)
                      for key, value in inputs.items()}
            t0 = time.monotonic()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(**inputs, **run_config["decoding"])
            token_ids = generated[0, length:].cpu().tolist()
            answer = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else eos
            truncated = len(token_ids) >= args.max_new_tokens and token_ids[-1] not in eos
            prediction, mode, error = None, None, None
            try:
                if truncated:
                    raise ValueError("generation_truncated")
                prediction, mode = parse_generated_answer(answer, inp, mapping)
            except ValueError as exc:
                error = str(exc)
            result = {"sample_id": row["sample_id"], "task": task,
                      "family": row["provenance"]["family"], "prompt": prompt,
                      "option_mapping": mapping, "raw_answer": answer,
                      "generated_token_ids": token_ids, "generated_tokens": len(token_ids),
                      "input_tokens": length, "generation_seconds": time.monotonic() - t0,
                      "truncated": truncated, "prediction": prediction, "parse_mode": mode,
                      "parse_error": error, "target": row["target"],
                      "scores": score_prediction(row, prediction)}
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            if (index + 1) % 10 == 0 or index + 1 == len(shard):
                status = {"rank": rank, "done": index + 1, "total": len(shard),
                          "elapsed_seconds": time.monotonic() - started}
                write_json(args.output / f"status-rank{rank:02d}.json", status)
                print(json.dumps(status), flush=True)
    if world == 1:
        summarize(args, selected, run_config)


if __name__ == "__main__":
    main()
