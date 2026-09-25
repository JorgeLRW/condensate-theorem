"""
Orthogonal Condensate PoC (Deterministic + Exact Tail Reduction)
=================================================================

Goal
----
Test the idea that tail handling can be a byproduct of orthogonal structure:
1) Build a rank-r orthogonal basis from important keys.
2) Route focus to high-subspace-energy tokens (keep set).
3) Condense remaining tail into grouped exact aggregates.
4) Reconstruct attention output exactly (floating-point tolerance).

For a fixed query q and tail groups C_g, define:
  Z_g = sum_{i in C_g} exp(s_i)
  N_g = sum_{i in C_g} exp(s_i) v_i
Then output is exactly:
  o = (sum_keep exp(s_i) v_i + sum_g N_g) / (sum_keep exp(s_i) + sum_g Z_g)

So grouping changes compute structure, not numerical meaning.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch


@dataclass
class Config:
    seq_len: int = 2048
    dim: int = 64
    rank_r: int = 16
    keep_topk_logits: int = 32
    keep_topk_grad: int = 32
    keep_window: int = 128
    keep_anchor: int = 1
    tail_groups: int = 16
    seed: int = 0
    fp16: bool = False
    deterministic: bool = False


def setup_determinism(enabled: bool) -> None:
    if not enabled:
        return
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def causal_logits(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    d = q.shape[-1]
    s = (q @ k.T) / math.sqrt(d)
    upper = torch.triu(torch.ones_like(s, dtype=torch.bool), diagonal=1)
    return s.masked_fill(upper, float("-inf"))


def full_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = causal_logits(q, k)
    p = torch.softmax(s, dim=-1)
    return p @ v, s


def grad_scores_for_query(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_idx: int,
) -> torch.Tensor:
    s = causal_logits(q, k).detach().clone().requires_grad_(True)
    p = torch.softmax(s, dim=-1)
    out = p @ v

    target = torch.zeros_like(out[query_idx])
    target[0] = 1.0
    loss = torch.nn.functional.mse_loss(out[query_idx], target)
    loss.backward()

    g = s.grad[query_idx].abs().detach()
    g[query_idx + 1:] = 0
    return g


def build_keep_mask(
    logits_q: torch.Tensor,
    grad_q: torch.Tensor,
    query_idx: int,
    keep_topk_logits: int,
    keep_topk_grad: int,
    keep_window: int,
    keep_anchor: int,
) -> torch.Tensor:
    n = query_idx + 1
    mask = torch.zeros(n, dtype=torch.bool, device=logits_q.device)

    if keep_anchor > 0:
        mask[:min(n, keep_anchor)] = True

    if keep_window > 0:
        ws = max(0, n - keep_window)
        mask[ws:n] = True

    k_logits = min(keep_topk_logits, n)
    if k_logits > 0:
        idx = torch.topk(logits_q[:n], k_logits).indices
        mask[idx] = True

    k_grad = min(keep_topk_grad, n)
    if k_grad > 0:
        idx = torch.topk(grad_q[:n], k_grad).indices
        mask[idx] = True

    return mask


def orthogonal_basis_from_keep(
    k_keep: torch.Tensor,
    rank_r: int,
) -> torch.Tensor:
    """
    Returns U [D, r_eff] orthonormal basis from keep keys.
    """
    d = k_keep.shape[-1]
    if k_keep.shape[0] == 0:
        return torch.eye(d, device=k_keep.device, dtype=k_keep.dtype)[:, :1]

    x = k_keep.T  # [D, K]
    q_mat, _ = torch.linalg.qr(x, mode="reduced")
    r_eff = min(rank_r, q_mat.shape[1])
    return q_mat[:, :r_eff]


def group_tail_by_orth_residual(
    k_tail: torch.Tensor,
    U: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    """
    Group tail tokens by residual energy after projection onto U.
    ids shape [N_tail] in [0, g-1].
    """
    n = k_tail.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=k_tail.device)

    proj = (k_tail @ U) @ U.T
    residual = ((k_tail - proj) ** 2).sum(dim=-1)  # [N_tail]

    g = min(groups, n)
    order = torch.argsort(residual)
    ids = torch.empty(n, dtype=torch.long, device=k_tail.device)
    for j in range(g):
        s = (j * n) // g
        e = ((j + 1) * n) // g
        ids[order[s:e]] = j
    return ids


def condensed_exact_output(
    logits_q: torch.Tensor,
    v_prefix: torch.Tensor,
    keep_mask: torch.Tensor,
    k_prefix: torch.Tensor,
    rank_r: int,
    tail_groups: int,
) -> tuple[torch.Tensor, int, int, float]:
    """
    Exact grouped reduction for one query.
    Returns output, kept_count, group_count, orth_explained_energy.
    """
    n = logits_q.shape[0]
    idx = torch.arange(n, device=logits_q.device)
    keep_idx = idx[keep_mask]
    tail_idx = idx[~keep_mask]

    k_keep = k_prefix[keep_idx] if keep_idx.numel() > 0 else k_prefix[:0]
    U = orthogonal_basis_from_keep(k_keep, rank_r)

    # Orth explained energy over all causal keys
    proj_all = (k_prefix @ U) @ U.T
    e_tot = (k_prefix ** 2).sum().clamp_min(1e-30)
    e_proj = (proj_all ** 2).sum()
    explained = (e_proj / e_tot).item()

    max_logit = torch.max(logits_q)
    z = torch.exp(logits_q - max_logit)  # [n]

    z_keep = z[keep_idx]
    numer_keep = (z_keep[:, None] * v_prefix[keep_idx]).sum(dim=0) if keep_idx.numel() > 0 else torch.zeros(
        v_prefix.shape[-1], device=v_prefix.device, dtype=v_prefix.dtype
    )
    denom_keep = z_keep.sum() if keep_idx.numel() > 0 else torch.tensor(0.0, device=v_prefix.device, dtype=v_prefix.dtype)

    if tail_idx.numel() == 0:
        out = numer_keep / (denom_keep + 1e-30)
        return out, keep_idx.numel(), 0, explained

    k_tail = k_prefix[tail_idx]
    ids = group_tail_by_orth_residual(k_tail, U, tail_groups)
    g = int(ids.max().item()) + 1

    numer_tail = torch.zeros_like(numer_keep)
    denom_tail = torch.tensor(0.0, device=v_prefix.device, dtype=v_prefix.dtype)

    for j in range(g):
        members = tail_idx[ids == j]
        z_j = z[members].sum()
        n_j = (z[members, None] * v_prefix[members]).sum(dim=0)
        numer_tail = numer_tail + n_j
        denom_tail = denom_tail + z_j

    out = (numer_keep + numer_tail) / (denom_keep + denom_tail + 1e-30)
    return out, keep_idx.numel(), g, explained


def run(cfg: Config) -> None:
    setup_determinism(cfg.deterministic)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if cfg.fp16 else torch.float32

    q = torch.randn(cfg.seq_len, cfg.dim, device=device, dtype=dtype)
    k = torch.randn(cfg.seq_len, cfg.dim, device=device, dtype=dtype)
    v = torch.randn(cfg.seq_len, cfg.dim, device=device, dtype=dtype)

    out_full, logits = full_attention(q, k, v)
    qi = cfg.seq_len - 1

    grad_q = grad_scores_for_query(q, k, v, qi)
    logits_q = logits[qi, : qi + 1]

    keep_mask = build_keep_mask(
        logits_q=logits_q,
        grad_q=grad_q,
        query_idx=qi,
        keep_topk_logits=cfg.keep_topk_logits,
        keep_topk_grad=cfg.keep_topk_grad,
        keep_window=cfg.keep_window,
        keep_anchor=cfg.keep_anchor,
    )

    out_cond, kept, groups, explained = condensed_exact_output(
        logits_q=logits_q,
        v_prefix=v[: qi + 1],
        keep_mask=keep_mask,
        k_prefix=k[: qi + 1],
        rank_r=cfg.rank_r,
        tail_groups=cfg.tail_groups,
    )

    ref = out_full[qi]
    abs_err = (ref - out_cond).abs()
    max_err = abs_err.max().item()
    mean_err = abs_err.mean().item()
    rel_err = (abs_err / (ref.abs() + 1e-8)).mean().item()

    total = qi + 1
    effective = kept + groups
    compression = total / max(1, effective)

    print("=" * 92)
    print("ORTHOGONAL CONDENSATE POC (Deterministic + Exact Tail Reduction)")
    print("=" * 92)
    print(
        f"device={device}  dtype={dtype}  deterministic={cfg.deterministic}  "
        f"seq_len={cfg.seq_len}  dim={cfg.dim}"
    )
    print(
        "keep="
        f"anchor({cfg.keep_anchor}) + window({cfg.keep_window}) + "
        f"topk_logits({cfg.keep_topk_logits}) + topk_grad({cfg.keep_topk_grad})"
    )
    print(f"rank_r={cfg.rank_r}  tail_groups={cfg.tail_groups}")
    print("-" * 92)
    print(f"query_idx                  : {qi}")
    print(f"original tokens            : {total}")
    print(f"kept explicit tokens       : {kept}")
    print(f"tail condensate groups     : {groups}")
    print(f"effective tokens processed : {effective}")
    print(f"compression factor         : {compression:.2f}x")
    print(f"orth explained energy      : {explained:.6f}")
    print("-" * 92)
    print(f"max abs error              : {max_err:.6e}")
    print(f"mean abs error             : {mean_err:.6e}")
    print(f"mean relative error        : {rel_err:.6e}")
    print(f"lossless (fp tolerance)    : {max_err < 1e-5}")
    print("=" * 92)


def parse() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--rank-r", type=int, default=16)
    p.add_argument("--keep-topk-logits", type=int, default=32)
    p.add_argument("--keep-topk-grad", type=int, default=32)
    p.add_argument("--keep-window", type=int, default=128)
    p.add_argument("--keep-anchor", type=int, default=1)
    p.add_argument("--tail-groups", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    a = p.parse_args()

    return Config(
        seq_len=a.seq_len,
        dim=a.dim,
        rank_r=a.rank_r,
        keep_topk_logits=a.keep_topk_logits,
        keep_topk_grad=a.keep_topk_grad,
        keep_window=a.keep_window,
        keep_anchor=a.keep_anchor,
        tail_groups=a.tail_groups,
        seed=a.seed,
        fp16=a.fp16,
        deterministic=a.deterministic,
    )


if __name__ == "__main__":
    run(parse())
