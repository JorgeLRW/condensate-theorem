"""
Phantom Weights Prototype (No Retraining)
=========================================

Idea:
  Quantize a linear weight W_fp -> W_q, then add a compact phantom correction
  DeltaW so inference uses W_eff = W_q + DeltaW.

This script fits DeltaW from calibration activations WITHOUT training the model.

Usage:
  python validation/phantom_weights.py
  python validation/phantom_weights.py --bits 2 --rank 16 --n-samples 4096
"""

import argparse
from dataclasses import dataclass

import torch


def quantize_symmetric_per_tensor(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Uniform symmetric quantization (dequantized output)."""
    assert bits >= 2
    qmax = (2 ** (bits - 1)) - 1

    max_abs = weight.abs().max().clamp(min=1e-8)
    scale = max_abs / qmax

    q = torch.round(weight / scale).clamp(-qmax, qmax)
    dequant = q * scale
    return dequant


def quantize_symmetric_per_channel(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-output-channel symmetric quantization (dequantized output)."""
    assert bits >= 2
    qmax = (2 ** (bits - 1)) - 1

    # scale per output channel (row)
    max_abs = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / qmax

    q = torch.round(weight / scale).clamp(-qmax, qmax)
    dequant = q * scale
    return dequant


def fit_full_delta_from_calibration(
    x_calib: torch.Tensor,
    y_target: torch.Tensor,
    y_quant: torch.Tensor,
    ridge: float = 1e-4,
) -> torch.Tensor:
    """
    Solve DeltaW (full-rank) in least squares:
      x @ DeltaW.T ~= (y_target - y_quant)

    x_calib: [N, in_dim]
    y_target: [N, out_dim] (fp output)
    y_quant: [N, out_dim] (quantized output)

    Returns DeltaW: [out_dim, in_dim]
    """
    residual = y_target - y_quant  # [N, out]
    x = x_calib  # [N, in]

    # Solve for B where x @ B ~= residual, with ridge regularization
    # B = (X^T X + lambda I)^-1 X^T R
    xt = x.transpose(0, 1)  # [in, N]
    xtx = xt @ x
    in_dim = xtx.shape[0]
    reg = ridge * torch.eye(in_dim, device=x.device, dtype=x.dtype)
    rhs = xt @ residual

    b = torch.linalg.solve(xtx + reg, rhs)  # [in, out]
    delta_w = b.transpose(0, 1).contiguous()  # [out, in]
    return delta_w


def low_rank_compress(delta_w: torch.Tensor, rank: int) -> torch.Tensor:
    """Truncated SVD compression of DeltaW."""
    out_dim, in_dim = delta_w.shape
    rank = max(1, min(rank, out_dim, in_dim))

    u, s, vh = torch.linalg.svd(delta_w, full_matrices=False)
    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]

    return (u_r * s_r.unsqueeze(0)) @ vh_r


@dataclass
class PhantomReport:
    mse_quant: float
    mse_phantom: float
    rel_improvement: float
    max_err_quant: float
    max_err_phantom: float


def evaluate_phantom(
    x_eval: torch.Tensor,
    w_fp: torch.Tensor,
    w_q: torch.Tensor,
    delta_w: torch.Tensor,
) -> PhantomReport:
    y_fp = x_eval @ w_fp.transpose(0, 1)
    y_q = x_eval @ w_q.transpose(0, 1)
    y_ph = x_eval @ (w_q + delta_w).transpose(0, 1)

    err_q = y_fp - y_q
    err_ph = y_fp - y_ph

    mse_q = (err_q.square().mean()).item()
    mse_ph = (err_ph.square().mean()).item()
    max_q = err_q.abs().max().item()
    max_ph = err_ph.abs().max().item()

    improvement = 0.0
    if mse_q > 0:
        improvement = 100.0 * (mse_q - mse_ph) / mse_q

    return PhantomReport(
        mse_quant=mse_q,
        mse_phantom=mse_ph,
        rel_improvement=improvement,
        max_err_quant=max_q,
        max_err_phantom=max_ph,
    )


def run_demo(
    bits: int,
    rank: int,
    n_samples: int,
    in_dim: int,
    out_dim: int,
    seed: int,
    per_channel: bool,
    ridge: float,
) -> None:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    w_fp = torch.randn(out_dim, in_dim, device=device, dtype=dtype) * (1.0 / (in_dim ** 0.5))
    if per_channel:
        w_q = quantize_symmetric_per_channel(w_fp, bits=bits)
    else:
        w_q = quantize_symmetric_per_tensor(w_fp, bits=bits)

    # Calibration/eval activations
    x_calib = torch.randn(n_samples, in_dim, device=device, dtype=dtype)
    x_eval = torch.randn(max(1024, n_samples // 2), in_dim, device=device, dtype=dtype)

    y_fp_calib = x_calib @ w_fp.transpose(0, 1)
    y_q_calib = x_calib @ w_q.transpose(0, 1)

    # Fit full delta, then compress to low-rank phantom delta
    delta_full = fit_full_delta_from_calibration(x_calib, y_fp_calib, y_q_calib, ridge=ridge)
    delta_rank = low_rank_compress(delta_full, rank=rank)

    report = evaluate_phantom(x_eval, w_fp, w_q, delta_rank)

    print("=" * 72)
    print("PHANTOM WEIGHTS DEMO (NO RETRAINING)")
    print("=" * 72)
    qmode = "per-channel" if per_channel else "per-tensor"
    print(f"device={device} bits={bits} qmode={qmode} rank={rank} in={in_dim} out={out_dim}")
    print(f"calibration_samples={n_samples} eval_samples={x_eval.shape[0]}")
    print(f"ridge={ridge}")
    print()
    print("Baseline quantized vs fp:")
    print(f"  mse:      {report.mse_quant:.6e}")
    print(f"  max|err|: {report.max_err_quant:.6e}")
    print()
    print("Phantom-corrected (W_q + DeltaW_rank):")
    print(f"  mse:      {report.mse_phantom:.6e}")
    print(f"  max|err|: {report.max_err_phantom:.6e}")
    print(f"  mse_gain: {report.rel_improvement:.2f}%")
    print()


def run_rank_sweep(
    bits: int,
    ranks: list[int],
    n_samples: int,
    in_dim: int,
    out_dim: int,
    seed: int,
    per_channel: bool,
    ridge: float,
) -> None:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    w_fp = torch.randn(out_dim, in_dim, device=device, dtype=dtype) * (1.0 / (in_dim ** 0.5))
    if per_channel:
        w_q = quantize_symmetric_per_channel(w_fp, bits=bits)
    else:
        w_q = quantize_symmetric_per_tensor(w_fp, bits=bits)

    x_calib = torch.randn(n_samples, in_dim, device=device, dtype=dtype)
    x_eval = torch.randn(max(1024, n_samples // 2), in_dim, device=device, dtype=dtype)

    y_fp_calib = x_calib @ w_fp.transpose(0, 1)
    y_q_calib = x_calib @ w_q.transpose(0, 1)
    delta_full = fit_full_delta_from_calibration(x_calib, y_fp_calib, y_q_calib, ridge=ridge)

    y_fp_eval = x_eval @ w_fp.transpose(0, 1)
    y_q_eval = x_eval @ w_q.transpose(0, 1)
    baseline_mse = (y_fp_eval - y_q_eval).square().mean().item()

    qmode = "per-channel" if per_channel else "per-tensor"
    print("=" * 72)
    print("PHANTOM WEIGHTS RANK SWEEP")
    print("=" * 72)
    print(f"device={device} bits={bits} qmode={qmode} in={in_dim} out={out_dim}")
    print(f"calibration_samples={n_samples} eval_samples={x_eval.shape[0]} ridge={ridge}")
    print(f"baseline_mse={baseline_mse:.6e}")
    print()
    print(f"{'rank':>6}  {'mse':>14}  {'gain%':>8}  {'max|err|':>12}")
    print("-" * 48)

    for rank in ranks:
        delta_rank = low_rank_compress(delta_full, rank=rank)
        report = evaluate_phantom(x_eval, w_fp, w_q, delta_rank)
        print(f"{rank:6d}  {report.mse_phantom:14.6e}  {report.rel_improvement:8.2f}  {report.max_err_phantom:12.6e}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--sweep-ranks", type=str, default="")
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--in-dim", type=int, default=1024)
    parser.add_argument("--out-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--ridge", type=float, default=1e-4)
    args = parser.parse_args()

    if args.sweep_ranks.strip():
        ranks = [int(x.strip()) for x in args.sweep_ranks.split(",") if x.strip()]
        run_rank_sweep(
            bits=args.bits,
            ranks=ranks,
            n_samples=args.n_samples,
            in_dim=args.in_dim,
            out_dim=args.out_dim,
            seed=args.seed,
            per_channel=args.per_channel,
            ridge=args.ridge,
        )
    else:
        run_demo(
            bits=args.bits,
            rank=args.rank,
            n_samples=args.n_samples,
            in_dim=args.in_dim,
            out_dim=args.out_dim,
            seed=args.seed,
            per_channel=args.per_channel,
            ridge=args.ridge,
        )
