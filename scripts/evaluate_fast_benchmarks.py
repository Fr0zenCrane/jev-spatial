"""Evaluate frozen image-only SAT/RefSpatial/VST requests using native, typed or unified models."""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_jev.batching import make_processor, prepare_batch  # noqa: E402
from spatial_jev.benchmark_metrics import summarize_benchmarks  # noqa: E402
from spatial_jev.generation_eval import parse_generated_answer  # noqa: E402
from spatial_jev.schema import compile_prompt  # noqa: E402
from spatial_jev.official_benchmarks import parse_choice, parse_points, summarize_official  # noqa: E402
from spatial_jev.robospatial_benchmarks import (  # noqa: E402
    parse_yes_no, summarize_robospatial, verify_official_aggregate,
)
from spatial_jev.unified import UnifiedBuilder, build_unified_model, predict_unified  # noqa: E402


def native_predict(model, processor, inp, config, device):
    prompt, mapping = compile_prompt(inp)
    task = inp["answer_space"]["kind"]
    suffix = {"choice": f"Answer with only the option number (1 to {len(mapping)}).",
              "scalar": "Answer with only the number and unit.",
              "point": "Answer with only one (x, y) pair."}[task]
    content = [{"type": "image"} for _ in inp["media"]]
    content.append({"type": "text", "text": prompt + "\n" + suffix})
    text = processor.apply_chat_template([{"role": "user", "content": content}],
                                         tokenize=False, add_generation_prompt=True)
    images = []
    for media in inp["media"]:
        with Image.open(media["uri"]) as image:
            images.append(image.convert("RGB"))
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
    length = inputs["input_ids"].shape[1]
    if length > config["max_sequence_length"]:
        raise ValueError("Input exceeds frozen token budget")
    inputs = {k: v.to(device=device, dtype=torch.bfloat16) if v.is_floating_point()
              else v.to(device) for k, v in inputs.items()}
    generated = model.generate(**inputs, max_new_tokens=128, do_sample=False, num_beams=1, use_cache=True)
    tokens = generated[0, length:].cpu().tolist()
    raw = processor.tokenizer.decode(tokens, skip_special_tokens=True)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    try:
        if len(tokens) >= 128 and tokens[-1] not in eos:
            raise ValueError("generation_truncated")
        prediction, mode = parse_generated_answer(raw, inp, mapping)
        return {"prediction": prediction, "raw_answer": raw, "parse_mode": mode,
                "output_tokens": len(tokens), "parse_error": None}
    except ValueError as exc:
        return {"prediction": None, "raw_answer": raw, "parse_error": str(exc),
                "output_tokens": len(tokens)}


def native_official_predict(model, processor, row, config, device):
    inp = row["input"]
    if row["benchmark"] == "robospatial_vq" and config.get("robospatial_vq_prompt") == "numbered":
        out = native_predict(model, processor, inp, config, device)
        out["robospatial_binary_format"] = "numbered_choices"
        return out
    content = [{"type": "image"} for _ in inp["media"]]
    # Molmo has a native HTML point format. Keep the benchmark request itself,
    # without forcing another model's tuple serialization; never use GT object metadata.
    prompt = inp["question"] if inp["answer_space"]["kind"] == "point" else row["native_prompt"]
    if inp["answer_space"]["kind"] == "point" and config.get("native_point_policy") in ("single", "two"):
        count = "one" if config["native_point_policy"] == "single" else "two"
        noun = "point" if count == "one" else "points"
        prompt = re.sub(r'\b(?:several|multiple|a few|some)\s+(?:points|spots|locations)\b',
                        f'{count} {noun}', prompt, flags=re.I)
        location = "location" if count == "one" else "locations"
        prompt += f"\nPoint to exactly {count} valid {location}. Use your usual coordinate output format."
    content.append({"type": "text", "text": prompt})
    text = processor.apply_chat_template([{"role": "user", "content": content}],
                                         tokenize=False, add_generation_prompt=True)
    images = []
    for media in inp["media"]:
        with Image.open(media["uri"]) as image:
            images.append(image.convert("RGB"))
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
    length = inputs["input_ids"].shape[1]
    if length > config["max_sequence_length"]:
        raise ValueError("Input exceeds official evaluation token budget")
    inputs = {k: v.to(device=device, dtype=torch.bfloat16) if v.is_floating_point()
              else v.to(device) for k, v in inputs.items()}
    generated = model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=1, use_cache=True)
    tokens = generated[0, length:].cpu().tolist()
    raw = processor.tokenizer.decode(tokens, skip_special_tokens=True)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    out = {"raw_answer": raw, "output_tokens": len(tokens), "input_tokens": length,
           "generation_truncated": len(tokens) >= 256 and tokens[-1] not in eos}
    try:
        # Author point scorers accept complete coordinate tuples present in a response,
        # including a response that hits its generation limit. Preserve this behavior.
        if inp["answer_space"]["kind"] == "choice":
            if row["benchmark"] == "robospatial_vq":
                prediction = parse_yes_no(raw)
                out.update(prediction=prediction, parse_mode="official_yes_no_prefix")
            else:
                prediction = parse_choice(raw, inp["answer_space"]["options"])
                out.update(prediction=prediction, parse_mode="letter_or_exact_option_text")
        else:
            points, mode = parse_points(raw, inp["media"][0]["width"], inp["media"][0]["height"])
            out.update(prediction=points[0], points=points, parse_mode=mode)
        out["parse_error"] = None
    except ValueError as exc:
        out.update(prediction=None, points=[], parse_error=str(exc))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["native", "typed", "unified"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--suite", type=Path, default=ROOT / "data/benchmarks/fast_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["fast", "official_images", "robospatial"], default="fast")
    parser.add_argument("--max-crops", type=int, help="Explicit recorded inference override")
    parser.add_argument("--native-point-policy", choices=["source", "single", "two"], default="source",
                        help="Source point cardinality or one-point control matching classifiers")
    parser.add_argument("--robospatial-vq-prompt", choices=["source", "numbered"], default="source",
                        help="Official Yes/No prompt or classifier-matched numbered-option control")
    parser.add_argument("--limit-per-benchmark", type=int, default=0, help="Smoke only; never official score")
    args = parser.parse_args()
    if args.limit_per_benchmark and args.limit_per_benchmark % 2:
        raise ValueError("Smoke limit must be even to retain SAT original/reverse pairs")
    if args.kind != "native" and not args.checkpoint:
        raise ValueError("Adapted model requires checkpoint")
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(4)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    config = json.loads((args.checkpoint / "config.json" if args.checkpoint else
                         ROOT / "configs/pilot_v0.json").read_text())
    config.update(base_model=str((ROOT / config["base_model"]).resolve()), preserve_option_order=True)
    if args.profile in ("official_images", "robospatial"):
        config.update(max_crops=24, max_sequence_length=64000)
    if args.max_crops is not None:
        config["max_crops"] = args.max_crops
    config["native_point_policy"] = args.native_point_policy
    config["robospatial_vq_prompt"] = args.robospatial_vq_prompt
    requests = [json.loads(line) for line in (args.suite / "requests.jsonl").read_text().splitlines()]
    if args.limit_per_benchmark:
        selected, counts = [], {}
        for row in requests:
            name = row["benchmark"]
            if counts.get(name, 0) < args.limit_per_benchmark:
                selected.append(row)
                counts[name] = counts.get(name, 0) + 1
        requests = selected
    run_config = {"kind": args.kind, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                  "base_revision": config["base_revision"], "max_crops": config["max_crops"],
                  "max_sequence_length": config["max_sequence_length"], "world_size": world,
                  "suite_sha256": hashlib.sha256((args.suite / "requests.jsonl").read_bytes()).hexdigest(),
                  "preserve_option_order": True, "n_requests": len(requests),
                  "limit_per_benchmark": args.limit_per_benchmark,
                  "profile": args.profile,
                  "native_max_new_tokens": 128 if args.profile == "fast" else 256,
                  "native_point_policy": args.native_point_policy,
                  "native_point_prompt_policy": ("benchmark question; native coordinate format; no GT object metadata"
                                                 if args.profile != "fast" else "fast singleton tuple request"),
                  "protocol": ("full official image test sets and scoring with local model adapters; "
                               "not a certified reproduction of Molmo2-ER paper inference" if args.profile != "fast"
                               else "matched fast image budget; not an exact reproduction of paper inference settings")}
    if args.profile == "robospatial":
        run_config.update(scorer_revision="c0095be6ca2d012086b3c141eccac4c879865cd4", num_points_to_match=2,
                          modality="RGB only", robospatial_vq_prompt=args.robospatial_vq_prompt,
                          native_vq_max_new_tokens=128 if args.robospatial_vq_prompt == "numbered" else 256)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    if world > 1:
        dist.barrier()
    processor = make_processor(config["base_model"], config["max_crops"])
    if args.kind == "unified":
        model = build_unified_model(config, args.checkpoint, trainable=False).to(device)
        builder = UnifiedBuilder(processor, config)
    elif args.kind == "typed":
        from spatial_jev.model import build_model
        model = build_model(config["base_model"], config, args.checkpoint, trainable=False).to(device)
    else:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            config["base_model"], trust_remote_code=True, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    start = time.perf_counter()
    predictions = []
    with (args.output / f"predictions-rank{rank:02d}.jsonl").open("x") as stream:
        for i, row in enumerate(requests[rank::world]):
            # The forward path sees only input. Scoring references are loaded after all predictions.
            inp = row["input"]
            task = inp["answer_space"]["kind"]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                if args.kind == "unified":
                    out = predict_unified(model, builder, inp, device, seed=3407)
                elif args.kind == "native":
                    out = (native_official_predict(model, processor, row, config, device)
                           if args.profile != "fast" else
                           native_predict(model, processor, inp, config, device))
                else:
                    # Batch helper's target is an inert placeholder; no benchmark label is read.
                    dummy = ({"option_id": inp["answer_space"]["options"][0]["id"]} if task == "choice"
                             else {"value": 0.0} if task == "scalar" else {"points": [[0.0, 0.0]]})
                    inputs, _, counts, mappings = prepare_batch(
                        [{"input": inp, "target": dummy}], processor, device, config, seed=None)
                    value = model(inputs, task, counts)
                    prediction = (mappings[0][int(value[0].argmax())] if task == "choice" else
                                  value[0].clamp(max=20).expm1().item() if task == "scalar" else
                                  value[0].cpu().tolist())
                    out = {"prediction": prediction}
            torch.cuda.synchronize()
            result = {"id": row["id"], "benchmark": row["benchmark"],
                      "seconds": time.perf_counter() - t0, **out}
            predictions.append(result)
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            if (i + 1) % 25 == 0:
                print(json.dumps({"rank": rank, "done": i + 1}), flush=True)
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, predictions)
        predictions = [r for shard in gathered for r in shard]
    if rank == 0:
        by_id = {r["id"]: r for r in predictions}
        if len(by_id) != len(predictions):
            raise ValueError("Duplicate predictions")
        (args.output / "predictions.jsonl").write_text("".join(
            json.dumps(by_id[r["id"]], allow_nan=False) + "\n" for r in requests))
        # Labels are first opened here, after inference is complete.
        references = {r["id"]: r for r in map(json.loads, (args.suite / "references.jsonl")
                                             .read_text().splitlines()) if r["id"] in by_id}
        if args.profile == "robospatial":
            summary = summarize_robospatial(requests, references, by_id)
            verify_official_aggregate(requests, references, by_id, args.output)
        elif args.profile == "official_images":
            summary = summarize_official(requests, references, by_id)
        else:
            summary = summarize_benchmarks(requests, references, by_id)
        result = {"config": run_config, "evaluation_wall_seconds": time.perf_counter() - start,
                  "metrics": summary}
        (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
