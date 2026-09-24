"""Native Molmo2 media processing with explicit readout targets."""

import random

import torch
from PIL import Image

from .schema import compile_prompt


def make_processor(base_path, max_crops=2):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(base_path), trust_remote_code=True, local_files_only=True, use_fast=False)
    # This remote processor predates the optional attribute introduced by Transformers 4.57.
    if not hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = None
    processor.image_processor.max_crops = max_crops
    processor.tokenizer.padding_side = "left"
    return processor


def prepare_batch(records, processor, device, config, seed=None):
    rng = random.Random(seed)
    tasks = {r["input"]["answer_space"]["kind"] for r in records}
    if len(tasks) != 1:
        raise ValueError("A microbatch must contain one task type")
    task = tasks.pop()
    texts, images, targets, counts, mappings = [], [], [], [], []
    for record in records:
        inp = record["input"]
        if len(inp["media"]) > config["max_images"]:
            raise ValueError("Image budget exceeded")
        order = None
        if task == "choice" and seed is not None:
            order = list(range(len(inp["answer_space"]["options"])))
            rng.shuffle(order)
        prompt, option_ids = compile_prompt(inp, order)
        content = [{"type": "image"} for _ in inp["media"]]
        content.append({"type": "text", "text": prompt})
        texts.append(processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
        for media in inp["media"]:
            with Image.open(media["uri"]) as image:
                images.append(image.convert("RGB"))
        if task == "choice":
            targets.append(option_ids.index(record["target"]["option_id"]))
            counts.append(len(option_ids))
        elif task == "scalar":
            targets.append(record["target"]["value"])
        else:
            targets.append(record["target"]["points"][0])
        mappings.append(option_ids)
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    if inputs["input_ids"].shape[1] > config["max_sequence_length"]:
        raise ValueError("Sequence budget exceeded; refusing to truncate media or question")
    inputs = {k: v.to(device=device, dtype=torch.bfloat16) if v.is_floating_point()
              else v.to(device) for k, v in inputs.items()}
    target = torch.tensor(targets, device=device, dtype=torch.long if task == "choice" else torch.float32)
    count_tensor = torch.tensor(counts, device=device) if task == "choice" else None
    return inputs, target, count_tensor, mappings
