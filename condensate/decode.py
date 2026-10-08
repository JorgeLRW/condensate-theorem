"""Layer-local sparse decode controller and paired dense/sparse survival metrics."""

import copy
import math

import torch
import torch.nn.functional as F

from .selector import (
    BLOCK_SIZE,
    WINDOW,
    block_count,
    block_ranges,
    build_decode_mask,
    select_blocks_kv_group,
    should_refresh,
)


def prefill(model, input_ids, chunk_size=1024):
    past = None
    with torch.inference_mode():
        for start in range(0, input_ids.shape[1], chunk_size):
            out = model(
                input_ids[:, start : start + chunk_size],
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
    return past, out.logits[0, -1].float()


def nll_for(logits, target):
    return float(-F.log_softmax(logits.float(), dim=-1)[int(target)].item())


def dense_reference(model, prompt, token_count):
    """Dense greedy continuation. Returns (prefilled cache, tokens, per-token NLL)."""
    past, logits = prefill(model, prompt)
    base_cache = copy.deepcopy(past)
    first = int(torch.argmax(logits).item())
    tokens = [first]
    losses = [nll_for(logits, first)]
    current = torch.tensor([[first]], device=prompt.device)
    with torch.inference_mode():
        for _ in range(1, token_count):
            out = model(current, past_key_values=past, use_cache=True)
            logits = out.logits[0, -1].float()
            nxt = int(torch.argmax(logits).item())
            tokens.append(nxt)
            losses.append(nll_for(logits, nxt))
            current = torch.tensor([[nxt]], device=prompt.device)
            past = out.past_key_values
    return base_cache, tokens, losses


class SparseDecodeController:
    """Replaces each layer's attention during decode with a per-head additive mask.

    The KV cache is not truncated; the mask alone decides which cached keys each head reads.
    """

    def __init__(self, model, apply_rope, window=WINDOW, block_size=BLOCK_SIZE):
        self.apply_rope = apply_rope
        self.window = window
        self.block_size = block_size
        self.active = False
        self.support = 97
        self.reuse = 1
        self.decode_step = 0
        self.selected = {}
        self.unique_position_history = []
        self._originals = []
        self._install(model)

    def _install(self, model):
        for layer in model.model.layers:
            module = layer.self_attn
            original = module.forward

            def wrapped(
                hidden_states,
                position_embeddings,
                attention_mask=None,
                past_key_values=None,
                cache_position=None,
                _original=original,
                _module=module,
                **kwargs,
            ):
                if self.active and hidden_states.shape[1] == 1 and past_key_values is not None:
                    attention_mask = self._make_mask(
                        _module, hidden_states, position_embeddings, attention_mask, past_key_values
                    )
                return _original(
                    hidden_states,
                    position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )

            module.forward = wrapped
            self._originals.append((module, original))

    def close(self):
        for module, original in self._originals:
            module.forward = original
        self._originals.clear()

    def reset(self, support, reuse):
        self.support = support
        self.reuse = reuse
        self.decode_step = 0
        self.selected = {}
        self.unique_position_history = []

    def _make_mask(self, module, hidden_states, position_embeddings, attention_mask, cache):
        layer_index = module.layer_idx
        if hidden_states.shape[0] != 1:
            raise ValueError("SparseDecodeController supports batch size 1 only")
        q_shape = hidden_states.shape[:-1]
        q = module.q_proj(hidden_states).view(*q_shape, -1, module.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, _ = self.apply_rope(q, q, cos, sin)
        queries = q[0, :, -1, :]
        keys = cache.layers[layer_index].keys[0]

        total_length = keys.shape[-2] + 1
        window_start, ranges = block_ranges(total_length, self.window, self.block_size)
        count = block_count(self.support, self.window, self.block_size)

        if should_refresh(self.decode_step, self.reuse, layer_index in self.selected):
            self.selected[layer_index] = select_blocks_kv_group(queries, keys, ranges, count)
        selected_by_head = self.selected[layer_index]

        group = queries.shape[0] // keys.shape[0]
        for kv in range(keys.shape[0]):
            group_selection = selected_by_head[kv * group : (kv + 1) * group]
            unique = {block for blocks in group_selection for block in blocks}
            self.unique_position_history.append(sum(ranges[i][1] - ranges[i][0] for i in unique))

        mask_dtype = attention_mask.dtype if attention_mask is not None else hidden_states.dtype
        mask = build_decode_mask(
            selected_by_head, ranges, window_start, total_length, mask_dtype, hidden_states.device
        )
        if attention_mask is None:
            return mask
        if attention_mask.ndim != 4:
            raise ValueError(f"expected a 4D attention mask, got shape {tuple(attention_mask.shape)}")
        return torch.minimum(mask, attention_mask.to(dtype=mask_dtype))


def evaluate_decode(model, base_cache, reference_tokens, reference_losses, controller, support, reuse):
    """Paired sparse-decode metrics against one dense greedy reference.

    Free-running pass: first exact divergence (Tdiv), one-based step index; the shared first
    token is excluded. Teacher-forced pass: per-step argmax agreement (TF) and NLL on the dense
    continuation, giving delta PPL against the dense model on the same tokens.
    """
    horizon = len(reference_tokens)
    device = base_cache.layers[0].keys.device

    controller.reset(support, reuse)
    controller.active = True
    auto_cache = copy.deepcopy(base_cache)
    current = torch.tensor([[reference_tokens[0]]], device=device)
    first_divergence = horizon
    try:
        with torch.inference_mode():
            for step in range(1, horizon):
                controller.decode_step = step
                out = model(current, past_key_values=auto_cache, use_cache=True)
                predicted = int(torch.argmax(out.logits[0, -1]).item())
                if first_divergence == horizon and predicted != reference_tokens[step]:
                    first_divergence = step
                current = torch.tensor([[predicted]], device=device)
                auto_cache = out.past_key_values
    finally:
        controller.active = False
    unique_positions = list(controller.unique_position_history)
    del auto_cache

    controller.reset(support, reuse)
    controller.active = True
    tf_cache = copy.deepcopy(base_cache)
    tf_matches = 0
    sparse_losses = []
    try:
        with torch.inference_mode():
            for step in range(1, horizon):
                controller.decode_step = step
                feed = torch.tensor([[reference_tokens[step - 1]]], device=device)
                out = model(feed, past_key_values=tf_cache, use_cache=True)
                logits = out.logits[0, -1].float()
                target = reference_tokens[step]
                tf_matches += int(torch.argmax(logits).item() == target)
                sparse_losses.append(nll_for(logits, target))
                tf_cache = out.past_key_values
    finally:
        controller.active = False

    dense_tail = reference_losses[1:]
    sparse_ppl = math.exp(sum(sparse_losses) / len(sparse_losses))
    dense_ppl = math.exp(sum(dense_tail) / len(dense_tail))
    steps = horizon - 1
    return {
        "first_divergence": first_divergence,
        "survives_horizon": first_divergence >= horizon,
        "survival_horizon": horizon,
        "teacher_forced_match": 100.0 * tf_matches / steps,
        "teacher_forced_steps": steps,
        "delta_ppl_percent": 100.0 * (sparse_ppl - dense_ppl) / dense_ppl,
        "mean_unique_distant_positions_per_kv": (
            float(sum(unique_positions) / len(unique_positions)) if unique_positions else 0.0
        ),
        "max_unique_distant_positions_per_kv": max(unique_positions, default=0),
    }
