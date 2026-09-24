"""Load a shared-classifier checkpoint and run its complete classification rollout."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_jev.batching import make_processor  # noqa: E402
from spatial_jev.unified import UnifiedBuilder, build_unified_model, predict_unified  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    torch.set_num_threads(4)
    config = json.loads((args.checkpoint / "config.json").read_text())
    value = json.loads(args.input.read_text())
    inp = value.get("input", value)
    seed = config["seed"] ^ int(value.get("sample_id", "0")[:8], 16)
    processor = make_processor(config["base_model"], config["max_crops"])
    model = build_unified_model(config, args.checkpoint, trainable=False).to(args.device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = predict_unified(model, UnifiedBuilder(processor, config), inp, args.device, seed)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
