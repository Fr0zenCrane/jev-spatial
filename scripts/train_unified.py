"""Matched-budget training/evaluation of the three shared-classifier pointing ablations."""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_jev.batching import make_processor  # noqa: E402
from spatial_jev.hierarchy import point_path, scalar_path  # noqa: E402
from spatial_jev.schema import TASKS  # noqa: E402
from spatial_jev.unified import (  # noqa: E402
    UnifiedBuilder, build_unified_model, collate_training, move_inputs, predict_unified,
    save_unified_checkpoint,
)
from train_pilot import read_records, utc_now  # noqa: E402


def summarize(rows):
    result = {}
    for task in TASKS:
        group = [r for r in rows if r["task"] == task]
        metric = {"n": len(group), "rollout_latency_p50_s": float(np.median([
            r["seconds"] for r in group]))}
        if task == "choice":
            metric["accuracy"] = float(np.mean([r["correct"] for r in group]))
        elif task == "scalar":
            metric.update(mae_m=float(np.mean([r["abs_error_m"] for r in group])),
                          median_abs_error_m=float(np.median([r["abs_error_m"] for r in group])),
                          abs_rel_positive=float(np.mean([r["relative_error"] for r in group
                                                          if r["relative_error"] is not None])),
                          within_10cm=float(np.mean([r["abs_error_m"] <= .1 for r in group])),
                          root_accuracy=float(np.mean([r["path"][0] == r["target_path"][0]
                                                       for r in group])),
                          overflow_predictions=sum(r["overflow"] for r in group))
        else:
            metric.update(mean_l2=float(np.mean([r["normalized_l2"] for r in group])),
                          hit_at_005=float(np.mean([r["normalized_l2"] <= .05 for r in group])),
                          hit_at_010=float(np.mean([r["normalized_l2"] <= .1 for r in group])),
                          mean_pixel_error=float(np.mean([r["pixel_error"] for r in group])),
                          prefix_accuracy=[float(np.mean([r["path"][:i] == r["target_path"][:i]
                                                          for r in group])) for i in [1, 2, 3]])
        result[task] = metric
    return result


@torch.inference_mode()
def evaluate(model, builder, pools, config, device, rank, world, limit):
    model.eval()
    rows = []
    for task in TASKS:
        examples = pools[task][:limit or None]
        for row in examples[rank::world]:
            seed = config["seed"] ^ int(row["sample_id"][:8], 16)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = predict_unified(model, builder, row["input"], device, seed)
            torch.cuda.synchronize()
            value = {"sample_id": row["sample_id"], "task": task,
                     "family": row["provenance"]["family"], "target": row["target"],
                     "seconds": time.perf_counter() - start, **out}
            if task == "choice":
                value["correct"] = out["prediction"] == row["target"]["option_id"]
            elif task == "scalar":
                error = abs(out["prediction"] - row["target"]["value"])
                value.update(abs_error_m=error, relative_error=(error / row["target"]["value"]
                             if row["target"]["value"] > 0 else None),
                             target_path=scalar_path(row["target"]["value"], builder.codebook),
                             overflow=out["path"][0] == builder.codebook["overflow_class"])
            else:
                dx = out["prediction"][0] - row["target"]["points"][0][0]
                dy = out["prediction"][1] - row["target"]["points"][0][1]
                image = row["input"]["media"][0]
                value.update(normalized_l2=math.hypot(dx, dy),
                             pixel_error=math.hypot(dx * image["width"], dy * image["height"]),
                             target_path=point_path(row["target"]["points"][0]))
            rows.append(value)
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        rows = [row for group in gathered for row in group]
    model.train()
    return summarize(rows), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/unified_v1.json")
    parser.add_argument("--variant", choices=["single_image", "crop_refill", "roi_mask"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--overfit-examples-per-task", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, help="Evaluate an existing checkpoint only")
    parser.add_argument("--resume", type=Path, help="Resume weights and optimizer at saved global step")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Initialize weights with the supplied new config and a fresh optimizer")
    parser.add_argument("--stop-at-unix", type=float,
                        help="Cooperatively stop all ranks and save before this UTC deadline")
    parser.add_argument("--full-eval", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if sum(bool(x) for x in [args.checkpoint, args.resume, args.init_checkpoint]) > 1:
        raise ValueError("Evaluation, resume and weight initialization are mutually exclusive")
    source = args.checkpoint or args.resume or args.init_checkpoint
    config_source = args.checkpoint or args.resume
    config = json.loads((config_source / "config.json" if config_source else args.config).read_text())
    if args.init_checkpoint:
        previous = json.loads((args.init_checkpoint / "config.json").read_text())
        for key in ["base_revision", "architecture", "max_choices", "lora_rank", "lora_alpha",
                    "lora_targets", "scalar_codebook", "point_depth", "point_variant"]:
            if previous[key] != config[key]:
                raise ValueError(f"Weight initialization contract mismatch: {key}")
        config["initialized_from"] = str(args.init_checkpoint.resolve())
        config["optimizer_policy"] = "fresh optimizer; weight initialization only"
    if source and config["point_variant"] != args.variant:
        raise ValueError("Checkpoint ablation mismatch")
    config["point_variant"] = args.variant
    config["base_model"] = str((ROOT / config["base_model"]).resolve())
    data = ROOT / config["data_directory"]
    config["data_summary_sha256"] = hashlib.sha256((data / "summary.json").read_bytes()).hexdigest()
    if args.max_steps:
        config["max_steps"] = args.max_steps
    if args.stop_at_unix:
        config["stop_at_unix"] = args.stop_at_unix
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(config["seed"])
    random.seed(config["seed"] + rank)
    train, dev = read_records(data / "train.jsonl"), read_records(data / "dev.jsonl")
    if args.overfit_examples_per_task:
        train = {task: rs[:args.overfit_examples_per_task] for task, rs in train.items()}
        dev = train
        config.update(global_batch_size=4, micro_batch_size=1, head_warmup_steps=5, warmup_steps=5,
                      validate_every=10, validation_examples_per_task=args.overfit_examples_per_task)
    config["overfit_examples_per_task"] = args.overfit_examples_per_task
    config["world_size"] = world
    micro = config["micro_batch_size"]
    if config["global_batch_size"] % (world * micro):
        raise ValueError("Global batch not divisible by worker count")
    accumulation = config["global_batch_size"] // (world * micro)
    config["gradient_accumulation_steps"] = accumulation
    if args.resume:
        config["resume_from"] = str(args.resume.resolve())
        config["resume_rng_policy"] = "rank seeds reinitialized; world-size migration is not bit-identical"
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (args.output / "environment.json").write_text(json.dumps({
            "started_at": utc_now(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
            "world_size": world, "evaluation": "internal dev; source overlap possible"}, indent=2))
        for pattern in ["src/spatial_jev/*.py", "scripts/*.py", "configs/*.json"]:
            for path in ROOT.glob(pattern):
                target = args.output / "source" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    if world > 1:
        dist.barrier()
    processor = make_processor(config["base_model"], config["max_crops"])
    builder = UnifiedBuilder(processor, config)
    base = build_unified_model(config, source, trainable=not bool(args.checkpoint)).to(device)
    torch.manual_seed(config["seed"] + rank)
    model = (DistributedDataParallel(base, device_ids=[device.index], find_unused_parameters=False)
             if world > 1 and not args.checkpoint else base)
    log = (args.output / "metrics.jsonl").open("a", buffering=1) if rank == 0 else None

    def emit(value):
        if rank == 0:
            value = {"time": utc_now(), **value}
            log.write(json.dumps(value, allow_nan=False) + "\n")
            print(json.dumps(value, allow_nan=False), flush=True)
            tmp = args.output / "status.tmp"
            tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
            tmp.replace(args.output / "status.json")

    if args.checkpoint:
        metrics, predictions = evaluate(base, builder, dev, config, device, rank, world, 0)
        if rank == 0:
            (args.output / "predictions.jsonl").write_text("".join(
                json.dumps(row, allow_nan=False) + "\n" for row in predictions))
            (args.output / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
        emit({"event": "evaluation_complete", "metrics": metrics})
    else:
        optimizer = torch.optim.AdamW([
            {"params": [p for p in base.backbone.parameters() if p.requires_grad],
             "lr": config["lora_learning_rate"], "weight_decay": config["weight_decay"]},
            {"params": list(base.classifier.parameters()), "lr": config["head_learning_rate"],
             "weight_decay": 0.0}], betas=tuple(config["adam_betas"]), eps=1e-8)
        start_step = 0
        if args.resume:
            state = torch.load(args.resume / "optimizer.pt", map_location="cpu", weights_only=True)
            start_step = int(state["step"])
            marker = json.loads((args.resume / "checkpoint.json").read_text())
            if marker["step"] != start_step or start_step >= config["max_steps"]:
                raise ValueError("Invalid resume step")
            optimizer.load_state_dict(state["optimizer"])
            del state
        emit({"event": "resumed" if args.resume else "initialized", "step": start_step,
              "world_size": world, "gradient_accumulation_steps": accumulation,
              "trainable_parameters": sum(
            p.numel() for p in base.parameters() if p.requires_grad),
            "classifier_parameters": sum(p.numel() for p in base.classifier.parameters())})
        metrics, _ = evaluate(base, builder, dev, config, device, rank, world,
                              config["validation_examples_per_task"])
        emit({"event": "validation", "step": start_step, "metrics": metrics})
        orders, start = {}, time.monotonic()
        done, stop_reason = start_step, "max_steps"
        deadline = config.get("stop_at_unix")
        training_started_unix = time.time()
        budget_progress = torch.zeros((), dtype=torch.float64, device=device)
        for step in range(start_step, config["max_steps"]):
            wall_progress = 0.0
            if deadline:
                # One shared clock keeps both LR and collective branches identical on all ranks.
                if rank == 0:
                    now = time.time()
                    budget_progress.fill_(1.0 if now >= deadline else
                                          (now - training_started_unix)
                                          / max(1, deadline - training_started_unix))
                if world > 1:
                    dist.broadcast(budget_progress, src=0)
                wall_progress = budget_progress.item()
                if wall_progress >= 1:
                    stop_reason = "wall_time_budget"
                    break
            task = TASKS[step % len(TASKS)]
            progress = step / config["max_steps"]
            if deadline and config.get("wall_time_lr_decay", False):
                progress = max(progress, wall_progress)
            decay = .1 + .9 * (1 + math.cos(math.pi * min(1, progress))) / 2
            head_warm = min(1.0, (step + 1) / config["warmup_steps"])
            lora_warm = min(1.0, max(0, step + 1 - config["head_warmup_steps"])
                            / config["warmup_steps"])
            optimizer.param_groups[0]["lr"] = config["lora_learning_rate"] * lora_warm * decay
            optimizer.param_groups[1]["lr"] = config["head_learning_rate"] * head_warm * decay
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            for part in range(accumulation):
                offset = ((step // len(TASKS) * accumulation + part) * world + rank) * micro
                examples = []
                for item in range(micro):
                    epoch, index = divmod(offset + item, len(train[task]))
                    key = task, epoch
                    if key not in orders:
                        order = list(range(len(train[task])))
                        random.Random(config["seed"] + epoch * 100 + TASKS.index(task)).shuffle(order)
                        orders[key] = order
                    row = train[task][orders[key][index]]
                    examples.append(builder.training_example(row, config["seed"] + offset + item))
                packed = collate_training(examples, processor.tokenizer.pad_token_id)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(move_inputs(packed["inputs"], device),
                                   packed["positions"].to(device), packed["counts"].to(device))
                    # Per-example mean prevents longer paths gaining extra task weight.
                    losses = torch.nn.functional.cross_entropy(
                        logits, packed["targets"].to(device), reduction="none")
                    loss = (losses * packed["loss_weights"].to(device)).sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite {task} loss at {step}")
                (loss / accumulation).backward()
                loss_sum += loss.detach() / accumulation
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in base.parameters() if p.requires_grad], config["gradient_clip"],
                error_if_nonfinite=True)
            optimizer.step()
            if world > 1:
                dist.all_reduce(loss_sum)
                loss_sum /= world
            done = step + 1
            if done == 1 or done % 10 == 0:
                emit({"event": "train", "step": done, "task": task, "loss": loss_sum.item(),
                      "grad_norm": norm.item(), "elapsed_seconds": time.monotonic() - start,
                      "lora_lr": optimizer.param_groups[0]["lr"],
                      "head_lr": optimizer.param_groups[1]["lr"],
                      "peak_memory_GB": torch.cuda.max_memory_allocated(device) / 1e9})
            if done % config["validate_every"] == 0 or done == config["max_steps"]:
                metrics, _ = evaluate(base, builder, dev, config, device, rank, world,
                                      config["validation_examples_per_task"])
                emit({"event": "validation", "step": done, "metrics": metrics})
            if done % config["save_every"] == 0 or done == config["max_steps"]:
                if world > 1:
                    dist.barrier()
                if rank == 0:
                    destination = args.output / f"checkpoint-{done:06d}"
                    save_unified_checkpoint(base, processor, config, destination, done, optimizer)
                    emit({"event": "checkpoint", "step": done, "path": str(destination)})
                if world > 1:
                    dist.barrier()
        if stop_reason == "wall_time_budget":
            if world > 1:
                dist.barrier()
            if rank == 0:
                destination = args.output / f"checkpoint-{done:06d}"
                if not (destination / "checkpoint.json").exists():
                    save_unified_checkpoint(base, processor, config, destination, done, optimizer)
                emit({"event": "checkpoint", "step": done, "path": str(destination),
                      "stop_reason": stop_reason})
            if world > 1:
                dist.barrier()
        if args.full_eval:
            metrics, predictions = evaluate(base, builder, dev, config, device, rank, world, 0)
            if rank == 0:
                (args.output / "predictions.jsonl").write_text("".join(
                    json.dumps(row, allow_nan=False) + "\n" for row in predictions))
                (args.output / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
            emit({"event": "full_evaluation", "metrics": metrics})
        emit({"event": "complete", "step": done, "stop_reason": stop_reason,
              "elapsed_seconds": time.monotonic() - start})
    if log:
        log.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
