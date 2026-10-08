"""Anchor + local window + query-dependent mean-pooled block selector (kv_group variant)."""

import math

import torch

WINDOW = 64
BLOCK_SIZE = 16


def block_count(support, window=WINDOW, block_size=BLOCK_SIZE):
    """Number of distant blocks that fit in a nominal support of `support` positions."""
    return max(0, support - 1 - window) // block_size


def block_ranges(total_length, window=WINDOW, block_size=BLOCK_SIZE):
    """Distant-block boundaries over positions [1, window_start), where window_start = total - window."""
    window_start = max(1, total_length - window)
    ranges = []
    start = 1
    while start < window_start:
        ranges.append((start, min(start + block_size, window_start)))
        start += block_size
    return window_start, ranges


def select_blocks_kv_group(queries, keys, ranges, count):
    """Pick `count` distant blocks per KV head and share each choice across its query-head group.

    queries: (query_heads, head_dim) post-RoPE query for the current token.
    keys:    (kv_heads, seq, head_dim) post-RoPE cached keys.
    Returns one list of block indices per query head.
    """
    query_heads, kv_heads = queries.shape[0], keys.shape[0]
    if query_heads % kv_heads:
        raise ValueError(f"query heads ({query_heads}) must be divisible by KV heads ({kv_heads})")
    group = query_heads // kv_heads
    if not ranges or count <= 0:
        return [[] for _ in range(query_heads)]

    centroids = torch.stack([keys[:, start:end].mean(dim=1) for start, end in ranges], dim=1)
    scale = math.sqrt(queries.shape[-1])
    take = min(count, len(ranges))

    by_kv = []
    for kv in range(kv_heads):
        group_query = queries[kv * group : (kv + 1) * group].mean(dim=0)
        scores = torch.mv(centroids[kv].float(), group_query.float()) / scale
        by_kv.append(torch.topk(scores, take).indices.tolist())
    return [by_kv[head // group] for head in range(query_heads)]


def should_refresh(decode_step, reuse, has_selection):
    """Distant blocks are reselected at one-based decode steps 1, 1+R, 1+2R, ..."""
    return not has_selection or reuse <= 1 or (decode_step - 1) % reuse == 0


def build_decode_mask(selected_by_head, ranges, window_start, total_length, dtype, device):
    """Additive mask of shape (1, query_heads, 1, total_length); 0 keeps a key, dtype-min drops it."""
    query_heads = len(selected_by_head)
    mask = torch.full(
        (1, query_heads, 1, total_length),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    mask[..., 0] = 0
    mask[..., window_start:total_length] = 0
    for head, block_indices in enumerate(selected_by_head):
        for index in block_indices:
            start, end = ranges[index]
            mask[0, head, 0, start:end] = 0
    return mask
