"""Portable Jev-Spatial merged-checkpoint API and command-line inference."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from .runtime import InferenceBuilder, SharedClassifier, UnifiedSpatialModel, predict_rollout
from .molmo2.modeling_molmo2 import Molmo2Model
from .molmo2.image_processing_molmo2 import Molmo2ImageProcessor
from .molmo2.video_processing_molmo2 import Molmo2VideoProcessor
from .molmo2.processing_molmo2 import Molmo2Processor


def load_processor(folder):
    """Load processor assets using installed code, with no Hub Python execution."""
    options = json.loads((folder / 'processor_config.json').read_text())
    options = {k: v for k, v in options.items() if k not in ('auto_map', 'processor_class')}
    return Molmo2Processor(
        image_processor=Molmo2ImageProcessor.from_pretrained(folder, local_files_only=True),
        video_processor=Molmo2VideoProcessor.from_pretrained(folder, local_files_only=True),
        tokenizer=AutoTokenizer.from_pretrained(folder, local_files_only=True, use_fast=False),
        chat_template=(folder / 'chat_template.jinja').read_text(),
        **options,
    )


def validate_input(inp, config):
    if not isinstance(inp, dict) or not isinstance(inp.get('question'), str) or not inp['question'].strip():
        raise ValueError('A nonempty question is required')
    media = inp.get('media', [])
    if not isinstance(media, list) or not 1 <= len(media) <= config['max_images']:
        raise ValueError(f"Expected 1 to {config['max_images']} images")
    for item in media:
        if not isinstance(item, dict) or item.get('kind') != 'image' or not isinstance(item.get('uri'), str):
            raise ValueError('Each media item requires kind=image and a local uri')
        if not Path(item['uri']).is_file():
            raise FileNotFoundError(item['uri'])
    space = inp.get('answer_space', {})
    task = space.get('kind')
    if task == 'choice':
        options = space.get('options', [])
        if not 2 <= len(options) <= config['max_choices']:
            raise ValueError(f"Expected 2 to {config['max_choices']} options")
        ids = [o.get('id') for o in options]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Option IDs must be unique nonempty strings')
        if any(not isinstance(o.get('text'), str) or not o['text'].strip() for o in options):
            raise ValueError('Every option needs nonempty text')
    elif task == 'scalar':
        if space.get('unit') != 'm' or not isinstance(space.get('quantity'), str):
            raise ValueError('This checkpoint estimates nonnegative lengths in meters')
    elif task == 'point':
        if len(media) != 1 or space.get('num_points') != 1 or space.get('coordinate_system') != 'normalized_xy':
            raise ValueError('Pointing requires one image and one normalized_xy point')
    else:
        raise ValueError('answer_space.kind must be choice, scalar, or point')


class JevSpatial:
    """One merged visual-language backbone and one shared classification head.

    Default prompting, candidate permutation, crop encoding, attention and KV-cache
    behavior match the original checkpoint's inference script. No PEFT dependency.
    """

    def __init__(self, model, processor, config, device):
        self.model = model
        self.processor = processor
        self.config = config
        self.device = torch.device(device)
        self.builder = InferenceBuilder(processor, config)

    @classmethod
    def from_pretrained(cls, model_path, device='cuda:0', max_crops=None,
                        max_sequence_length=None, preserve_option_order=None):
        folder = Path(model_path).expanduser().resolve()
        if not (folder / 'jev_config.json').is_file():
            raise FileNotFoundError(f'Expected an exported Jev-Spatial bundle at {folder}')
        config = json.loads((folder / 'jev_config.json').read_text())
        if config.get('format') != 'jev-spatial-merged/v1':
            raise ValueError('Unsupported release format')
        if max_crops is not None:
            if max_crops < 1:
                raise ValueError('max_crops must be positive')
            config['max_crops'] = max_crops
        if max_sequence_length is not None:
            config['max_sequence_length'] = max_sequence_length
        if preserve_option_order is not None:
            config['preserve_option_order'] = preserve_option_order
        # Native Molmo2 code is installed with this package, separately from HF weights.
        dtype = {'bfloat16': torch.bfloat16, 'float32': torch.float32}.get(config['runtime_dtype'])
        if dtype is None:
            raise ValueError('Unsupported runtime dtype')
        backbone, loading = Molmo2Model.from_pretrained(
            folder / 'backbone', local_files_only=True,
            dtype=dtype, attn_implementation='sdpa', output_loading_info=True)
        if any(loading.get(key) for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
            raise ValueError(f'Checkpoint load was not exact: {loading}')
        classifier = SharedClassifier(backbone.config.text_config.hidden_size, config['max_choices'])
        classifier.load_state_dict(load_file(str(folder / 'classifier.safetensors')))
        model = UnifiedSpatialModel(backbone, classifier).to(device).requires_grad_(False).eval()
        processor = load_processor(folder / 'processor')
        if not hasattr(processor, 'audio_tokenizer'):
            processor.audio_tokenizer = None
        processor.image_processor.max_crops = config['max_crops']
        processor.tokenizer.padding_side = 'left'
        return cls(model, processor, config, device)

    def predict(self, value, seed=None):
        """Return the original rollout record, including IDs, path and per-stage logits."""
        inp = value.get('input', value)
        validate_input(inp, self.config)
        if seed is None:
            seed = self.config['seed'] ^ int(value.get('sample_id', '0')[:8], 16)
        with torch.inference_mode(), torch.autocast(
                self.device.type, dtype=torch.bfloat16,
                enabled=self.device.type == 'cuda' and self.config.get('runtime_dtype', 'bfloat16') == 'bfloat16'):
            return predict_rollout(self.model, self.builder, inp, self.device, seed)

    @staticmethod
    def _media(images):
        if isinstance(images, (str, Path)):
            images = [images]
        return [{'kind': 'image', 'uri': str(image)} for image in images]

    def classify(self, images, question, options, seed=None):
        choices = [{'id': f'option_{i}', 'text': text} for i, text in enumerate(options)]
        return self.predict({'media': self._media(images), 'question': question,
                             'answer_space': {'kind': 'choice', 'options': choices}}, seed=seed)

    def measure(self, images, question, quantity='height', seed=None):
        return self.predict({'media': self._media(images), 'question': question,
                             'answer_space': {'kind': 'scalar', 'quantity': quantity, 'unit': 'm'}}, seed=seed)

    def point(self, image, question, seed=None):
        return self.predict({'media': self._media(image), 'question': question,
                             'answer_space': {'kind': 'point', 'coordinate_system': 'normalized_xy',
                                              'num_points': 1}}, seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Local merged checkpoint directory')
    parser.add_argument('--input', type=Path, required=True, help='One input JSON or a request JSONL')
    parser.add_argument('--output', type=Path, help='Write JSON/JSONL to this file, otherwise stdout')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, help='Override the original per-sample seed rule')
    parser.add_argument('--max-crops', type=int)
    parser.add_argument('--max-sequence-length', type=int)
    parser.add_argument('--preserve-option-order', action='store_true', default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = JevSpatial.from_pretrained(args.model, args.device, args.max_crops,
                                      args.max_sequence_length, args.preserve_option_order)
    is_jsonl = args.input.suffix == '.jsonl'
    values = ([json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
              if is_jsonl else [json.loads(args.input.read_text())])
    # CLI media paths are relative to the request file; the Python API uses the caller's cwd.
    for value in values:
        for item in value.get('input', value).get('media', []):
            if not Path(item['uri']).is_absolute():
                item['uri'] = str((args.input.parent / item['uri']).resolve())
    results = [model.predict(value, seed=args.seed) for value in values]
    text = ''.join(json.dumps(result, allow_nan=False) + '\n' for result in results)
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
