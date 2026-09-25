"""
Gradient Condensate PoC (Exact Tail Aggregation)
================================================

Idea tested:
1) Use gradient signal (-G^T) to define which keys are important.
2) Keep important keys explicitly.
3) Condense tail keys into rank-r groups.
4) Replace each tail group with ONE synthetic condensate token:
      exp(s*) = sum_i exp(s_i)
      v*      = sum_i exp(s_i) v_i / sum_i exp(s_i)

For a fixed query, this replacement is mathematically exact for softmax attention:
    sum_i softmax(s)_i v_i
is unchanged up to floating-point roundoff.

This PoC validates that exactness and reports the effective token count reduction.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch


@dataclass
class PoCConfig:
    seq_len: int = 2048
    dim: int = 64
    tail_groups: int = 16
    keep_topk: int = 64
    keep_window: int = 128
    keep_anchor: int = 1
    seed: int = 0
    dtype: torch.dtype = torch.float32


def causal_logits(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """q,k: [T,D] -> logits [T,T] causal-masked."""
    d = q.shape[-1]
    logits = (q @ k.T) / math.sqrt(d)
    mask = torch.triu(torch.ones_like(logits, dtype=torch.bool), diagonal=1)
    logits = logits.masked_fill(mask, float("-inf"))
    return logits


def full_attention_output(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = causal_logits(q, k)
    probs = torch.softmax(logits, dim=-1)
    out = probs @ v
    return out, logits


def query_gradient_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_idx: int,
) -> torch.Tensor:
    """
    Returns gradient-based key importance for one query:
        importance_i = |(-G^T)_{i,query_idx}| = |dL/dS_{query_idx,i}|
    """
    logits = causal_logits(q, k)
    logits = logits.clone().detach().requires_grad_(True)

    probs = torch.softmax(logits, dim=-1)
    out = probs @ v

    target = torch.zeros_like(out[query_idx])
    target[0] = 1.0
    loss = torch.nn.functional.mse_loss(out[query_idx], target)
    loss.backward()

    g = logits.grad
    scores = g[query_idx].abs().detach()  # [T]
    scores[query_idx + 1 :] = 0
    return scores


def build_keep_set(
    logits_q: torch.Tensor,
    grad_scores_q: torch.Tensor,
    query_idx: int,
    keep_topk: int,
    keep_window: int,
    keep_anchor: int,
) -> torch.Tensor:
    """Union of anchor + window + topk(logit) + topk(gradient)."""
    device = logits_q.device
    keep_mask = torch.zeros(query_idx + 1, dtype=torch.bool, device=device)

    if keep_anchor > 0:
        keep_mask[: min(keep_anchor, query_idx + 1)] = True

    if keep_window > 0:
        ws = max(0, query_idx - keep_window + 1)
        keep_mask[ws : query_idx + 1] = True

    k1 = min(keep_topk, query_idx + 1)
    if k1 > 0:
        top_logits = torch.topk(logits_q[: query_idx + 1], k1).indices
        keep_mask[top_logits] = True

        top_grad = torch.topk(grad_scores_q[: query_idx + 1], k1).indices
        keep_mask[top_grad] = True

    return keep_mask


def assign_tail_groups(grad_tail: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Gradient-space grouping in 1D: sort by grad score and split into equal bins.
    Returns group ids [N_tail] in [0, groups-1].
    """
    n = grad_tail.numel()
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=grad_tail.device)

    g = min(groups, n)
    order = torch.argsort(grad_tail)
    ids = torch.empty(n, dtype=torch.long, device=grad_tail.device)

    for j in range(g):
        s = (j * n) // g
        e = ((j + 1) * n) // g
        ids[order[s:e]] = j

    return ids


def condensed_query_output(
    logits_q: torch.Tensor,
    v: torch.Tensor,
    keep_mask: torch.Tensor,
    grad_scores_q: torch.Tensor,
    tail_groups: int,
) -> tuple[torch.Tensor, int, int]:
    """
    Exact condensation for one query.

    For each tail group C:
        z_C = sum_{i in C} exp(logits_i)
        u_C = sum_{i in C} exp(logits_i) v_i / z_C

    Then softmax over kept tokens + condensates is exactly equivalent
    to original softmax over all tokens (up to fp roundoff).
    """
    q_len = logits_q.shape[0]
    idx = torch.arange(q_len, device=logits_q.device)

    keep_idx = idx[keep_mask]
    tail_idx = idx[~keep_mask]

    max_logit = torch.max(logits_q)
    exp_all = torch.exp(logits_q - max_logit)

    z_keep = exp_all[keep_idx]  # [K]
    numer_keep = z_keep[:, None] * v[keep_idx]  # [K,D]

    if tail_idx.numel() == 0:
        z_total = z_keep.sum()
        out = numer_keep.sum(dim=0) / z_total
        return out, keep_idx.numel(), 0

    grad_tail = grad_scores_q[tail_idx]
    group_ids = assign_tail_groups(grad_tail, tail_groups)
    g = int(group_ids.max().item()) + 1

    z_groups = []
    u_groups = []

    for j in range(g):
        members = tail_idx[group_ids == j]
        z_j = exp_all[members].sum()
        u_j = (exp_all[members, None] * v[members]).sum(dim=0) / (z_j + 1e-30)
        z_groups.append(z_j)
        u_groups.append(u_j)

    z_groups_t = torch.stack(z_groups, dim=0)  # [G]
    u_groups_t = torch.stack(u_groups, dim=0)  # [G,D]

    numer = numer_keep.sum(dim=0) + (z_groups_t[:, None] * u_groups_t).sum(dim=0)
    z_total = z_keep.sum() + z_groups_t.sum()
    out = numer / (z_total + 1e-30)

    return out, keep_idx.numel(), g


def run_poc(cfg: PoCConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = torch.randn(cfg.seq_len, cfg.dim, device=device, dtype=cfg.dtype)
    k = torch.randn(cfg.seq_len, cfg.dim, device=device, dtype=cfg.dtype)
    v = torch.randn(cfg.seq_len, cfg.dim, device=device, dtype=cfg.dtype)

    out_full, logits = full_attention_output(q, k, v)
    query_idx = cfg.seq_len - 1

    grad_scores = query_gradient_scores(q, k, v, query_idx=query_idx)

    logits_q = logits[query_idx, : query_idx + 1]
    keep_mask = build_keep_set(
        logits_q=logits_q,
        grad_scores_q=grad_scores,
        query_idx=query_idx,
        keep_topk=cfg.keep_topk,
        keep_window=cfg.keep_window,
        keep_anchor=cfg.keep_anchor,
    )

    out_cond, kept, groups = condensed_query_output(
        logits_q=logits_q,
        v=v,
        keep_mask=keep_mask,
        grad_scores_q=grad_scores,
        tail_groups=cfg.tail_groups,
    )

    ref = out_full[query_idx]
    abs_err = (ref - out_cond).abs()
    max_err = abs_err.max().item()
    mean_err = abs_err.mean().item()
    rel = (abs_err / (ref.abs() + 1e-8)).mean().item()

    total = query_idx + 1
    effective = kept + groups
    compression = total / max(1, effective)

    print("=" * 86)
    print("GRADIENT CONDENSATE POC (Exact Tail Aggregation)")
    print("=" * 86)
    print(f"device={device}  dtype={cfg.dtype}  seq_len={cfg.seq_len}  dim={cfg.dim}")
    print(
        f"keep: topk={cfg.keep_topk}, window={cfg.keep_window}, anchor={cfg.keep_anchor} | "
        f"tail_groups={cfg.tail_groups}"
    )
    print("-" * 86)
    print(f"query_idx                  : {query_idx}")
    print(f"original tokens            : {total}")
    print(f"kept explicit tokens       : {kept}")
    print(f"tail condensate groups     : {groups}")
    print(f"effective tokens processed : {effective}")
    print(f"compression factor         : {compression:.2f}x")
    print("-" * 86)
    print(f"max abs error              : {max_err:.6e}")
    print(f"mean abs error             : {mean_err:.6e}")
    print(f"mean relative error        : {rel:.6e}")
    print(f"lossless (fp tolerance)    : {max_err < 1e-5}")
    print("=" * 86)


def parse_args() -> PoCConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--tail-groups", type=int, default=16)
    parser.add_argument("--keep-topk", type=int, default=64)
    parser.add_argument("--keep-window", type=int, default=128)
    parser.add_argument("--keep-anchor", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    return PoCConfig(
        seq_len=args.seq_len,
        dim=args.dim,
        tail_groups=args.tail_groups,
        keep_topk=args.keep_topk,
        keep_window=args.keep_window,
        keep_anchor=args.keep_anchor,
        seed=args.seed,
        dtype=torch.float16 if args.fp16 else torch.float32,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_poc(cfg)
