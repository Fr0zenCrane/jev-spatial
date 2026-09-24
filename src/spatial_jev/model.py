"""Molmo2 multimodal backbone with supervised typed readouts."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def last_valid_hidden(hidden, mask):
    positions = torch.arange(mask.shape[1], device=mask.device)[None].expand_as(mask)
    indices = positions.masked_fill(~mask.bool(), -1).max(-1).values
    if (indices < 0).any():
        raise ValueError("An input has no valid tokens")
    return hidden[torch.arange(len(hidden), device=hidden.device), indices]


class DecisionHeads(nn.Module):
    def __init__(self, hidden_size, max_choices=32, middle=256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.choice = nn.Linear(hidden_size, max_choices)
        self.scalar = nn.Sequential(nn.Linear(hidden_size, middle), nn.GELU(), nn.Linear(middle, 1))
        self.point = nn.Sequential(nn.Linear(hidden_size, middle), nn.GELU(), nn.Linear(middle, 2))

    def forward(self, hidden, task, option_counts=None):
        hidden = self.norm(hidden.float())
        if task == "choice":
            logits = self.choice(hidden)
            invalid = torch.arange(logits.shape[1], device=logits.device)[None] >= option_counts[:, None]
            return logits.masked_fill(invalid, -torch.inf)
        if task == "scalar":
            return F.softplus(self.scalar(hidden).squeeze(-1))  # log1p(meters)
        if task == "point":
            return self.point(hidden).sigmoid()
        raise ValueError(task)


class SpatialDecisionModel(nn.Module):
    def __init__(self, backbone, heads):
        super().__init__()
        self.backbone = backbone
        self.heads = heads

    def forward(self, inputs, task, option_counts=None):
        output = self.backbone(**inputs, use_cache=False)
        hidden = last_valid_hidden(output.last_hidden_state, inputs["attention_mask"])
        # Keep the small heads and losses in FP32 while the backbone uses BF16.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.heads(hidden.float(), task, option_counts)


def build_model(base_path, config, checkpoint=None, trainable=True):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText

    full = AutoModelForImageTextToText.from_pretrained(
        str(base_path), trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    backbone = full.model
    hidden_size = full.config.text_config.hidden_size
    del full  # Retain multimodal processing, discard unused vocabulary readout.
    if checkpoint:
        backbone = PeftModel.from_pretrained(
            backbone, str(Path(checkpoint) / "adapter"), is_trainable=trainable)
    else:
        lora = LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                          lora_dropout=config["lora_dropout"],
                          target_modules=config["lora_targets"], bias="none")
        backbone = get_peft_model(backbone, lora)
    if trainable and config.get("gradient_checkpointing", True):
        backbone.get_base_model().gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    heads = DecisionHeads(hidden_size, config["max_choices"], config["head_hidden_size"])
    if checkpoint:
        from safetensors.torch import load_file
        heads.load_state_dict(load_file(str(Path(checkpoint) / "heads.safetensors")))
    if not any("lora_" in n and p.requires_grad for n, p in backbone.named_parameters()) and trainable:
        raise ValueError("No trainable LoRA parameters were attached")
    if any(p.requires_grad for n, p in backbone.named_parameters() if "vision_backbone" in n):
        raise ValueError("Vision/projector parameters must remain frozen")
    model = SpatialDecisionModel(backbone, heads)
    if not trainable:
        model.requires_grad_(False)
        model.eval()
    return model


def task_loss(prediction, target, task, config):
    if task == "choice":
        return F.cross_entropy(prediction, target.long())
    if task == "scalar":
        return F.smooth_l1_loss(prediction, target.float().log1p(),
                                beta=config["scalar_huber_beta"])
    if task == "point":
        return F.smooth_l1_loss(prediction, target.float(), beta=config["point_huber_beta"])
    raise ValueError(task)


def save_checkpoint(model, processor, config, directory, step, optimizer=None):
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.backbone.save_pretrained(directory / "adapter", safe_serialization=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.heads.state_dict().items()},
              str(directory / "heads.safetensors"))
    processor.save_pretrained(directory / "processor")
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if optimizer is not None:
        torch.save({"step": step, "optimizer": optimizer.state_dict()}, directory / "optimizer.pt")
    (directory / "checkpoint.json").write_text(json.dumps({
        "step": step, "base_model": config["base_model"], "base_revision": config["base_revision"],
        "format": "base_plus_lora_plus_heads", "calibrated": False,
    }, indent=2) + "\n")
