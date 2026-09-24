"""Single-GPU or torchrun DDP training of Molmo2-ER typed decision heads."""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spatial_jev.batching import make_processor, prepare_batch  # noqa: E402
from spatial_jev.model import build_model, save_checkpoint, task_loss  # noqa: E402
from spatial_jev.schema import TASKS, validate_record  # noqa: E402


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_records(path):
    pools = {task: [] for task in TASKS}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            validate_record(row)
            pools[row["input"]["answer_space"]["kind"]].append(row)
    if not all(pools.values()):
        raise ValueError("All three tasks must have examples")
    return pools


@torch.inference_mode()
def evaluate(model, processor, pools, config, device, rank=0, world=1):
    model.eval()
    result = {}
    for task in TASKS:
        sums = torch.zeros(4, dtype=torch.float64, device=device)
        selected = pools[task][:config["validation_examples_per_task"]]
        for row in selected[rank::world]:
            # SAT can put the correct option first. A fixed, per-example permutation
            # prevents validation from rewarding that source formatting shortcut.
            evaluation_seed = config["seed"] ^ int(row["sample_id"][:8], 16)
            inputs, target, counts, _ = prepare_batch(
                [row], processor, device, config, seed=evaluation_seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(inputs, task, counts)
                loss = task_loss(prediction, target, task, config)
            sums[0] += 1
            sums[1] += loss
            if task == "choice":
                sums[2] += (prediction.argmax(-1) == target).float().sum()
            elif task == "scalar":
                physical = prediction.clamp(max=20).expm1()
                error = (physical - target).abs()
                sums[2] += error.sum()
                sums[3] += (error / target.clamp_min(1e-4)).sum()
            else:
                error = (prediction - target).norm(dim=-1)
                sums[2] += error.sum()
                sums[3] += (error <= 0.05).float().sum()
        if world > 1:
            dist.all_reduce(sums)
        count, total_loss, primary, secondary = sums.cpu().tolist()
        result[task] = {"n": int(count), "loss": total_loss / count}
        if task == "choice":
            result[task]["accuracy"] = primary / count
        elif task == "scalar":
            result[task].update({"mae_m": primary / count, "abs_rel_eps_1e4": secondary / count})
        else:
            result[task].update({"normalized_l2": primary / count,
                                 "coordinate_hit_at_0.05": secondary / count})
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot_v0.json"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--overfit-examples-per-task", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HOME", str(root / ".hf_cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = json.loads(args.config.read_text())
    config["base_model"] = str((root / config["base_model"]).resolve())
    config["data_directory"] = str(args.data.resolve())
    config["data_summary_sha256"] = hashlib.sha256((args.data / "summary.json").read_bytes()).hexdigest()
    if args.max_steps:
        config["max_steps"] = args.max_steps
    if args.overfit_examples_per_task:
        config["global_batch_size"] = 4
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    seed = config["seed"]
    random.seed(seed + rank)
    torch.manual_seed(seed)  # Identical head initialization on every rank.
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    global_batch, micro = config["global_batch_size"], config["micro_batch_size"]
    if global_batch % (world * micro):
        raise ValueError("Global batch must be divisible by world size * micro batch")
    accumulation = global_batch // (world * micro)
    train = read_records(args.data / "train.jsonl")
    dev = read_records(args.data / "dev.jsonl")
    if args.overfit_examples_per_task:
        train = {task: rows[:args.overfit_examples_per_task] for task, rows in train.items()}
        dev = train
        config["validation_examples_per_task"] = args.overfit_examples_per_task
        config["head_warmup_steps"] = min(10, config["head_warmup_steps"])
        config["warmup_steps"] = min(10, config["warmup_steps"])
    config["overfit_examples_per_task"] = args.overfit_examples_per_task
    config["world_size"] = world
    config["gradient_accumulation_steps"] = accumulation
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        environment = {
            "started_at": utc_now(), "torch": torch.__version__,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
            "gpu": torch.cuda.get_device_name(device), "world_size": world,
            "benchmark_status": "not_selected; internal development set only",
        }
        (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
        (args.output / "source.diff").write_text(
            subprocess.check_output(["git", "diff"], cwd=root, text=True))
        snapshot = args.output / "source"
        for pattern in ["src/spatial_jev/*.py", "scripts/*.py", "configs/*.json", "pyproject.toml"]:
            for path in root.glob(pattern):
                target = snapshot / path.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    if world > 1:
        dist.barrier()
    processor = make_processor(config["base_model"], config["max_crops"])
    base_model = build_model(config["base_model"], config).to(device)
    torch.manual_seed(seed + rank)
    model = DistributedDataParallel(base_model, device_ids=[local_rank],
                                    find_unused_parameters=True) if world > 1 else base_model
    optimizer = torch.optim.AdamW([
        {"params": [p for p in base_model.backbone.parameters() if p.requires_grad],
         "lr": config["lora_learning_rate"], "weight_decay": config["weight_decay"]},
        {"params": list(base_model.heads.parameters()),
         "lr": config["head_learning_rate"], "weight_decay": 0.0},
    ], betas=tuple(config["adam_betas"]), eps=1e-8)
    log = (args.output / "metrics.jsonl").open("a", buffering=1) if rank == 0 else None

    def emit(value):
        if rank == 0:
            value = {"time": utc_now(), **value}
            log.write(json.dumps(value) + "\n")
            print(json.dumps(value), flush=True)
            temporary = args.output / "status.json.tmp"
            temporary.write_text(json.dumps(value, indent=2) + "\n")
            temporary.replace(args.output / "status.json")

    emit({"event": "initialized", "trainable_parameters": sum(
        p.numel() for p in base_model.parameters() if p.requires_grad)})
    validation = evaluate(base_model, processor, dev, config, device, rank, world)
    emit({"event": "validation", "step": 0, "metrics": validation})
    orders = {}
    start = time.monotonic()
    model.train()
    for step in range(config["max_steps"]):
        task = TASKS[step % len(TASKS)]
        decay = 0.1 + 0.9 * (1 + math.cos(math.pi * step / config["max_steps"])) / 2
        head_warm = min(1.0, (step + 1) / config["warmup_steps"])
        lora_warm = min(1.0, max(0, step + 1 - config["head_warmup_steps"]) / config["warmup_steps"])
        optimizer.param_groups[0]["lr"] = config["lora_learning_rate"] * lora_warm * decay
        optimizer.param_groups[1]["lr"] = config["head_learning_rate"] * head_warm * decay
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for part in range(accumulation):
            offset = ((step // len(TASKS) * accumulation + part) * world + rank) * micro
            examples = []
            for j in range(micro):
                epoch, index = divmod(offset + j, len(train[task]))
                key = task, epoch
                if key not in orders:
                    order = list(range(len(train[task])))
                    random.Random(seed + epoch * 100 + TASKS.index(task)).shuffle(order)
                    orders[key] = order
                examples.append(train[task][orders[key][index]])
            inputs, target, counts, _ = prepare_batch(
                examples, processor, device, config, seed=seed + offset + part)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(inputs, task, counts)
                loss = task_loss(prediction, target, task, config) * config["loss_weights"][task]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {task} loss at step {step}")
            (loss / accumulation).backward()
            loss_sum += loss.detach() / accumulation
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], config["gradient_clip"],
            error_if_nonfinite=True)
        optimizer.step()
        if world > 1:
            dist.all_reduce(loss_sum)
            loss_sum /= world
        completed = step + 1
        if completed == 1 or completed % 10 == 0:
            emit({"event": "train", "step": completed, "task": task,
                  "loss": loss_sum.item(), "grad_norm": norm.item(),
                  "lora_lr": optimizer.param_groups[0]["lr"],
                  "head_lr": optimizer.param_groups[1]["lr"],
                  "elapsed_seconds": time.monotonic() - start,
                  "peak_memory_GB": torch.cuda.max_memory_allocated(device) / 1e9})
        if completed % config["validate_every"] == 0 or completed == config["max_steps"]:
            validation = evaluate(base_model, processor, dev, config, device, rank, world)
            emit({"event": "validation", "step": completed, "metrics": validation})
        if completed % config["save_every"] == 0 or completed == config["max_steps"]:
            if world > 1:
                dist.barrier()
            if rank == 0:
                directory = args.output / f"checkpoint-{completed:06d}"
                save_checkpoint(base_model, processor, config, directory, completed, optimizer)
                emit({"event": "checkpoint", "step": completed, "path": str(directory.resolve())})
            if world > 1:
                dist.barrier()
    emit({"event": "complete", "step": config["max_steps"],
          "elapsed_seconds": time.monotonic() - start})
    if log:
        log.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
