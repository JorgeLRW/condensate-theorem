# Attention-Mass Condensation for Sparse Decoding

Reference implementation of the selector, metrics, and survival sweep from
*Attention-Mass Condensation for Sparse Decoding: Margin Stability, Query-Dependent Retrieval, and Operating Limits*
(Jorge L. Ruiz Williams).

The paper's central claim is conditional, not a blanket claim that trained attention can be sparsified.
Sparse attention preserves the dense greedy decision only if:

1. attention concentrates on a small, query-dependent support, **and**
2. the omitted contribution stays inside the downstream decision margin.

On Qwen2-0.5B, the second condition fails at the supports tested here, so none of the 60 paired runs stays identical to dense decoding through 128 tokens. See [Results](#results).

## Contents

| Path | What it is |
|---|---|
| [condensate/selector.py](condensate/selector.py) | Block budget, block ranges, KV-group selector, refresh rule, and additive decode mask |
| [condensate/decode.py](condensate/decode.py) | Layer-local sparse decode controller, dense reference, and paired survival metrics |
| [condensate/data.py](condensate/data.py) | WikiText-2 prefix construction |
| [scripts/run_survival.py](scripts/run_survival.py) | Survival sweep CLI; defaults reproduce the paper's 60-run grid |
| [tests/test_selector.py](tests/test_selector.py) | CPU unit checks: block budget, sharing, refresh cadence, mask, omitted-mass identity |
| [validate.py](validate.py) | Runs the unit tests; with `--smoke`, reruns one paired case and compares it to the archive |
| [results/qwen_survival_clean.json](results/qwen_survival_clean.json) | The 60-run archived sweep behind the paper's survival table |
| [results/rerun_2048_prompt0.json](results/rerun_2048_prompt0.json) | Rerun of the 2K / prompt 0 / S=97,769 case from this repository |

## Method

### Omitted-mass identity

Let `C` be the retained keys and `D` the omitted keys, with `ε = a(D)` the dense attention mass on `D`.
Let `o_C` and `o_D` be value averages normalized within `C` and `D`. Then

```
o_full = (1 − ε) o_C + ε o_D
Δo = o_sparse − o_full = ε (o_C − o_D)
‖Δo‖₂ ≤ 2 ε V_max
```

Retained mass alone does not control `Δo`: the omitted value directions and the size of `o_C − o_D` also matter.

### Downstream margin condition

Let `δ = Σ_l κ_l E_l`, where `E_l = (Σ_h (2 ε_{l,h} V_{max,l,h})²)^{1/2}` and `κ_l` is a layer sensitivity constant.
If `2δ < γ`, where `γ` is the dense top-1 logit margin, the greedy argmax is preserved.
This is a sufficient condition, not a necessary one. The `κ_l` are not practically bounded, so the condition is not
checked by this code.

### Selector (kv_group variant)

Each decode step, for each layer, the selector keeps:

- the anchor (position 0),
- the last `W = 64` positions,
- `r` distant blocks of `M = 16` positions, chosen per KV head.

Distant blocks are scored by `s_b = q·μ_b / √d`, where `μ_b` is the mean of the post-RoPE cached keys in block `b`
and `q` is the mean post-RoPE query of the query-head group that shares the KV head.
The selection is shared across that group.

Nominal supports and block counts:

| Support `S` | Distant blocks `r` | Positions per KV head (anchor + window + blocks) |
|---|---|---|
| 97 | 2 | 1 + 64 + 32 |
| 193 | 8 | 1 + 64 + 128 |
| 385 | 20 | 1 + 64 + 320 |
| 769 | 44 | 1 + 64 + 704 |

With reuse `R`, distant blocks are reselected at one-based decode steps `1, 1+R, 1+2R, …`. The paper's sweep uses `R = 1`.

The cache is not truncated. The controller adds a per-head additive mask to each layer's attention during 1-token decode steps.
Anchor, window, and selected blocks are kept; every other cached key is masked.

The paper also specifies `shared`, `head_union`, `rerank`, and `per_head` selector variants. This repository implements only `kv_group`,
which is the variant evaluated in the survival sweep.

### Metrics

Both the dense and sparse runs start from the same dense-prefilled cache and the same first token.

- **Survival**: whether the free-running sparse greedy output matches the dense greedy output exactly for 128 tokens.
  `Tdiv` is the one-based step of the first mismatch.
- **Teacher-forced match (TF)**: sparse argmax agreement with the dense tokens, teacher-forced on the dense continuation, excluding the shared first token.
- **ΔPPL**: `100 · (exp(mean sparse NLL) − exp(mean dense NLL)) / exp(mean dense NLL)`, computed on the dense continuation
  (teacher-forced, not free-running, not held-out perplexity).

## Results

These are the paper's headline numbers for Qwen2-0.5B: 3 contexts (2K, 8K, 16K) × 4 supports (97, 193, 385, 769) × 5 WikiText-2 prefixes per context, with `R = 1` and 128 greedy tokens (60 paired runs). The full per-run table is in the paper's `tab:decode_survival` and in [results/qwen_survival_clean.json](results/qwen_survival_clean.json).

- **Survival**: 0 of 60 runs match dense decoding exactly through 128 tokens.
- **Distributional quality**: for `S ≥ 193`, 7 of 9 context-support cells have median teacher-forced ΔPPL within 5% of dense.
  Prompt-level ranges include severe 16K outliers: up to +9877% at `S = 97` and +152% at `S = 769`.
- **Warning regime**: all 7 runs with TF below 70% have ΔPPL above +100%. These come from two prefixes (0 and 4, at 16K).
  This suggests a warning regime, not a validated threshold.

The archive includes 15 prefixes and 4 supports, so the 60 runs are paired measurements over 15 prefixes, not 60 independent prompts.

### Smoke rerun from this repository

`python validate.py --smoke` reruns 2K / prompt 0 / `S = 97, 769` and compares the results to the archive.
On the reference machine (RTX 4090 Laptop GPU, 16 GB, torch 2.6.0+cu124, transformers 4.57.1, float16, SDPA) every compared field matches the archived row.
The rerun takes about 5 minutes, including the dense reference.

| Support | Tdiv | TF | ΔPPL |
|---|---|---|---|
| 97 | 2 | 93.70% | +5.07% |
| 769 | 22 | 98.43% | −0.34% |

Hardware or library differences can change float16 results. A `DIFF` line from `validate.py` means the rerun differs from the archive beyond the tolerance (1e-3); it is not a failure of the paper's claim.

## Reproducing

Requirements: a CUDA GPU with enough memory for Qwen2-0.5B in float16, Python 3.10+, and the packages in [requirements.txt](requirements.txt).

```bash
pip install -r requirements.txt

# CPU unit checks (selector, mask, omitted-mass identity)
python validate.py

# GPU: rerun one paired case and compare to the archive
python validate.py --smoke

# GPU: the paper's full 60-run sweep (writes JSON after every row)
python scripts/run_survival.py --output results/survival_rerun.json

# If a run stops partway, continue it from the saved rows (same settings required)
python scripts/run_survival.py --resume --output results/survival_rerun.json
```

The CLI defaults match the paper: `Qwen/Qwen2-0.5B`, contexts `2048,8192,16384`, supports `97,193,385,769`, `--reuse 1`, `--prompts 5`, `--max-new-tokens 128`.
Model and dataset are downloaded on first use.

## Limitations

- One small model (Qwen2-0.5B). Mistral-7B was only smoke-tested for mechanics and is not characterized.
- Five prefixes per context. The results do not establish a population failure rate or a universal sparse-attention limit.
- The selector is intentionally simple. Mean pooling can dilute isolated high-scoring keys. The paper has not separated coarse retrieval error, insufficient support,
  harmful omitted value directions, and recursive cache drift as causes of failure.
- **Timing is not claimed.** The paper's headline operator timings used a proprietary optimized Triton kernel that is not in this repository.
  Those timings exclude discovery, are not matched-quality, and are not end-to-end serving speed. This repository makes no timing claim.
- The paper's retrieval grid, reuse results, and H2O comparison use separate protocols and are not reproduced here.
- Earlier GPT-2 oracle scripts and a benchmark CSV that did not match the paper were removed. They remain available in git history.

## Citation

See [CITATION.cff](CITATION.cff).

## License

Code in this repository is released under the [MIT License](LICENSE).
The `LICENSE` file also notes that the proprietary optimized Triton kernel is not included and is licensed separately.
