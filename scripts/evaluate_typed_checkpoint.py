"""Full pilot dev predictions for a saved three-head checkpoint; no training."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_jev.batching import make_processor, prepare_batch  # noqa: E402
from spatial_jev.model import build_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = json.loads((args.checkpoint / "config.json").read_text())
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.json").write_text(json.dumps({
            "checkpoint": str(args.checkpoint.resolve()), "training_config": config,
            "data": str(args.data.resolve()), "world_size": world,
            "evaluation": "full internal dev, fixed per-example option permutation"}, indent=2))
    if world > 1:
        dist.barrier()
    processor = make_processor(config["base_model"], config["max_crops"])
    model = build_model(config["base_model"], config, args.checkpoint, trainable=False).to(device)
    source = [json.loads(line) for line in (args.data / "dev.jsonl").read_text().splitlines()]
    predictions = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in source[rank::world]:
            task = row["input"]["answer_space"]["kind"]
            seed = config["seed"] ^ int(row["sample_id"][:8], 16)
            inputs, _, counts, mappings = prepare_batch([row], processor, device, config, seed=seed)
            output = model(inputs, task, counts)
            if task == "choice":
                prediction = mappings[0][int(output[0].argmax())]
                probabilities = output[0, :len(mappings[0])].softmax(-1).cpu().tolist()
            elif task == "scalar":
                prediction = output[0].clamp(max=20).expm1().item()
                probabilities = None
            else:
                prediction = output[0].cpu().tolist()
                probabilities = None
            predictions.append({"sample_id": row["sample_id"], "task": task,
                                "family": row["provenance"]["family"], "target": row["target"],
                                "prediction": prediction, "mapping": mappings[0],
                                "probabilities": probabilities})
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, predictions)
        predictions = [r for shard in gathered for r in shard]
    if rank == 0:
        by_id = {r["sample_id"]: r for r in predictions}
        if len(by_id) != len(predictions) or set(by_id) != {r["sample_id"] for r in source}:
            raise ValueError("Prediction coverage mismatch")
        ordered = [by_id[r["sample_id"]] for r in source]
        (args.output / "predictions.jsonl").write_text("".join(
            json.dumps(r, allow_nan=False) + "\n" for r in ordered))
        print(json.dumps({"state": "complete", "n": len(ordered),
                          "checkpoint": str(args.checkpoint)}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
