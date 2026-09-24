"""Experimental same-scene inference with shared vision KV and isolated question branches."""

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .hierarchy import ROOT_BOX, child_box, decode_point, decode_scalar
from .runtime import causal_visual_mask, move_inputs


def branch_mask(prefix_length, owners, positions, query_start, device, visual=None, stages=None):
    """A question reads the image prefix and its own causal history, never other questions."""
    owners = torch.tensor(owners, device=device)
    positions = torch.tensor(positions, device=device)
    same = owners[query_start:, None] == owners[None, :]
    past = positions[None, :] <= positions[query_start:, None]
    if visual is not None:
        visual = torch.tensor(visual, dtype=torch.bool, device=device)
        stages = torch.tensor(stages, device=device)
        past |= (visual[query_start:, None] & visual[None, :]
                 & (stages[query_start:, None] == stages[None, :]))
    allowed = same & past
    allowed[:, :prefix_length] = True
    return torch.zeros((1, 1, len(owners) - query_start, len(owners)), device=device).masked_fill(
        ~allowed[None, None], -torch.inf)


class ScenePrefix:
    """One physical KV cache, with logical positions restarted for each question suffix."""

    def __init__(self, backbone, builder, images, rendered, device):
        self.backbone, self.builder, self.device = backbone, builder, device
        self.dtype = next(backbone.parameters()).dtype
        encoded, _ = builder.encode(rendered[0], images)
        types = encoded.pop('token_type_ids')[0].tolist()
        self.length = max(i for i, kind in enumerate(types) if kind) + 1
        self.prefix_types = types[:self.length]
        self.suffixes = []
        for text in rendered:
            for _ in images:
                if not text.startswith('<|image|>'):
                    raise ValueError('Expected all images before the question')
                text = text[len('<|image|>'):]
            tail, _ = builder.encode(text, [], continuation=True)
            self.suffixes.append(tail['input_ids'])
        if not torch.equal(encoded['input_ids'][:, self.length:], self.suffixes[0]):
            raise ValueError('Prefix split changed question tokenization')
        encoded['input_ids'] = encoded['input_ids'][:, :self.length]
        encoded['attention_mask'] = causal_visual_mask(self.length, self.prefix_types, [0] * self.length)
        inputs = move_inputs(encoded, device, self.dtype)
        output = backbone(**inputs, use_cache=True)
        self.cache = output.past_key_values
        self.owners = [-1] * self.length
        self.positions = list(range(self.length))
        self.visual = list(self.prefix_types)
        self.stages = [0] * self.length
        self.next_positions = {}

    def forward(self, branches, stage=0):
        """Evaluate text suffixes or one-token decode steps in one packed forward."""
        start = len(self.owners)
        ends, token_parts, logical, media_parts = [], [], [], {}
        for owner, value in branches.items():
            encoded = value if isinstance(value, dict) else {'input_ids': value}
            ids = encoded['input_ids']
            n = ids.shape[1]
            position = self.next_positions.get(owner, self.length)
            self.owners.extend([owner] * n)
            self.positions.extend(range(position, position + n))
            self.visual.extend(encoded['token_type_ids'][0].tolist() if 'token_type_ids' in encoded else [0] * n)
            self.stages.extend([stage] * n)
            self.next_positions[owner] = position + n
            logical.extend(range(position, position + n))
            token_parts.append(ids.to(self.device))
            ends.append(len(self.owners) - start - 1)
            for key in ['pixel_values', 'image_token_pooling', 'image_grids', 'image_num_crops']:
                if key in encoded:
                    media_parts.setdefault(key, []).append(encoded[key])
        media = move_inputs({key: torch.cat(parts) for key, parts in media_parts.items()}, self.device, self.dtype)
        mask = branch_mask(self.length, self.owners, self.positions, start, self.device, self.visual, self.stages)
        output = self.backbone(
            input_ids=torch.cat(token_parts, dim=1), attention_mask=mask,
            position_ids=torch.tensor([logical], device=self.device),
            cache_position=torch.arange(start, len(self.owners), device=self.device),
            past_key_values=self.cache, use_cache=True, **media)
        self.cache = output.past_key_values
        return output.last_hidden_state[0, ends]


def compact_cache(cache, batch_index, indices):
    return [(layer.keys[batch_index:batch_index+1].index_select(2, indices),
             layer.values[batch_index:batch_index+1].index_select(2, indices)) for layer in cache.layers]


def refine_batch(model, branches, caches, past_types, past_stages, stage, device, dtype, pad_id):
    """Batch independent crop continuations without dense attention between branches."""
    owners = list(branches)
    past_lengths = [len(past_types[i]) for i in owners]
    lengths = [branches[i]['input_ids'].shape[1] for i in owners]
    pmax, qmax = max(past_lengths), max(lengths)
    layer_count = len(caches[owners[0]])
    cache = DynamicCache((
        torch.cat([F.pad(caches[i][layer][0], (0, 0, pmax - p, 0)) for i, p in zip(owners, past_lengths)]),
        torch.cat([F.pad(caches[i][layer][1], (0, 0, pmax - p, 0)) for i, p in zip(owners, past_lengths)]))
        for layer in range(layer_count))
    mask = torch.full((len(owners), 1, qmax, pmax + qmax), -torch.inf)
    ids = torch.full((len(owners), qmax), pad_id, dtype=torch.long)
    positions = torch.zeros_like(ids)
    media = {}
    for b, (owner, p, q) in enumerate(zip(owners, past_lengths, lengths)):
        encoded = branches[owner]
        types = encoded['token_type_ids'][0].tolist()
        past_types[owner].extend(types)
        past_stages[owner].extend([stage] * q)
        local_mask = causal_visual_mask(p + q, past_types[owner], past_stages[owner], p)[0, 0]
        mask[b, 0, :q, pmax-p:pmax] = local_mask[:, :p]
        mask[b, 0, :q, pmax:pmax+q] = local_mask[:, p:]
        mask[b, 0, q:, pmax-p] = 0
        ids[b, :q] = encoded['input_ids'][0]
        positions[b, :q] = torch.arange(p, p + q)
        for key in ['pixel_values', 'image_token_pooling', 'image_grids', 'image_num_crops']:
            if key in encoded:
                media.setdefault(key, []).append(encoded[key])
    inputs = move_inputs({key: torch.cat(values) for key, values in media.items()}, device, dtype)
    out = model.backbone(input_ids=ids.to(device), attention_mask=mask.to(device),
                         position_ids=positions.to(device), past_key_values=cache,
                         cache_position=torch.arange(pmax, pmax + qmax, device=device),
                         use_cache=True, **inputs)
    hidden = out.last_hidden_state[torch.arange(len(owners), device=device),
                                   torch.tensor(lengths, device=device) - 1]
    if stage == 1:
        for b, (owner, p, q) in enumerate(zip(owners, past_lengths, lengths)):
            indices = torch.cat([torch.arange(pmax-p, pmax, device=device),
                                 torch.arange(pmax, pmax+q, device=device)])
            caches[owner] = compact_cache(out.past_key_values, b, indices)
    return hidden


@torch.inference_mode()
def predict_scene(runner, inputs, seed=3407):
    """Share the original image and batch each decision level, preserving separate crop paths."""
    if not inputs:
        return []
    media = [m['uri'] for m in inputs[0]['media']]
    if any([m['uri'] for m in inp['media']] != media for inp in inputs):
        raise ValueError('All questions must use the same ordered image set')
    builder, model, device = runner.builder, runner.model, runner.device
    if builder.variant != 'crop_refill':
        raise ValueError('This experimental runner supports the released crop-refill checkpoint')
    images = builder.load_images(inputs[0])
    initial = [builder.initial(inp, seed) for inp in inputs]
    shared = ScenePrefix(model.backbone, builder, images, [v[0] for v in initial], device)
    results = [{'path': [], 'logits': [], 'mapping': v[2], 'roi_support_counts': []} for v in initial]
    boxes, counts = [ROOT_BOX] * len(inputs), [v[1] for v in initial]
    pending = dict(enumerate(shared.suffixes))
    caches, past_types, past_stages = {}, {}, {}
    for stage in range(3):
        owners = list(pending)
        if stage == 0:
            hidden = shared.forward(pending)
        else:
            hidden = refine_batch(model, pending, caches, past_types, past_stages, stage,
                                  device, shared.dtype, builder.processor.tokenizer.pad_token_id)
        logits = model.classifier(hidden, torch.tensor([counts[i] for i in owners], device=device)).float().cpu()
        pending = {}
        for index, value in zip(owners, logits):
            inp, result = inputs[index], results[index]
            task, count = inp['answer_space']['kind'], counts[index]
            choice = int(value[:count].argmax())
            result['path'].append(choice)
            result['logits'].append(value[:count].tolist())
            result['input_tokens'] = shared.next_positions[index] if stage == 0 else len(past_types[index])
            if task == 'choice' or stage == 2 or (task == 'scalar' and (
                    stage == 1 or choice in (0, len(builder.codebook['edges_m'])))):
                continue
            if task == 'point':
                boxes[index] = child_box(boxes[index], choice)
                question, counts[index] = builder.point_prompt(inp, boxes[index], stage + 1), 9
                extra = [builder.crop(images[0], boxes[index])]
            else:
                question, counts[index] = builder.scalar_prompt(inp, choice)
                extra = []
            text = f'{choice+1}<|im_end|>\n' + builder.render(question, len(extra))
            encoded, _ = builder.encode(text, extra, continuation=True)
            if result['input_tokens'] + encoded['input_ids'].shape[1] > runner.config['max_sequence_length']:
                raise ValueError('Rollout sequence budget exceeded')
            pending[index] = encoded
            if stage == 0:
                indices = torch.tensor([j for j, owner in enumerate(shared.owners) if owner in (-1, index)], device=device)
                caches[index] = compact_cache(shared.cache, 0, indices)
                past_types[index] = shared.prefix_types + [0] * shared.suffixes[index].shape[1]
                past_stages[index] = [0] * result['input_tokens']
        if not pending:
            break
    for inp, result in zip(inputs, results):
        task, path = inp['answer_space']['kind'], result['path']
        result['prediction'] = (result['mapping'][path[0]] if task == 'choice' else
                                decode_scalar(path, builder.codebook) if task == 'scalar' else decode_point(path))
    return results
