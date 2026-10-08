# Attention-Mass Condensation for Sparse Decoding

Reference implementation of the Condensation Principle from
*Attention-Mass Condensation for Sparse Decoding: Margin Stability, Query-Dependent Retrieval, and Operating Limits*
(Jorge L. Ruiz Williams).

## The principle

Sparse attention can preserve the dense greedy decision when two conditions hold:

1. a trained model's attention is sufficiently concentrated on a small, query-dependent subset, and
2. the contribution of everything omitted stays inside the model's downstream decision margin.

"Condensation" refers only to attention mass concentrating onto a small support. The principle does not claim that attention is
universally sparse, that a fixed support size is always sufficient, or that a particular retrieval rule recovers the right support.

## Theory

### 1. Omitted-mass identity (exact in real arithmetic)

Let `C` be the retained keys, `D` the omitted keys, and `ε = a(D)` the dense attention mass on `D`.
Let `o_C` and `o_D` be the value averages normalized within `C` and `D`. Then

```
o_full = (1 − ε) o_C + ε o_D
Δo = o_sparse − o_full = ε (o_C − o_D)
‖Δo‖₂ ≤ 2 ε V_max
```

Retained mass alone does not control `Δo`: the omitted value directions and the size of `o_C − o_D` also matter.

### 2. Downstream margin condition (sufficient)

For each layer `l`, let `E_l = (Σ_h (2 ε_{l,h} V_{max,l,h})²)^{1/2}` over the heads `h`, and let `κ_l` bound the gain from that
layer's attention output to the final logits. With `δ = Σ_l κ_l E_l`, if `2δ < γ`, where `γ` is the dense top-1 logit margin,
then sparse and dense attention have the same greedy argmax.

The condition is sufficient, not necessary. The `κ_l` are not practically bounded in a real transformer, so this repository does not check it at runtime.

### What the theory does not give

- Sparse and dense outputs are not bit-identical, because removing nonzero softmax terms changes the partition function.
  The question is greedy decision stability, not IEEE-754 equality.
- A matching teacher-forced decision does not guarantee a matching free-running trajectory. The margin condition has to hold on the states the sparse model actually produces.
- Failing the margin condition does not imply that the argmax changes.

## Selector (kv_group variant)

Each decode step, for each layer, the selector keeps:

- the anchor (position 0),
- the last `W = 64` positions,
- `r` distant blocks of `M = 16` positions, chosen per KV head.

Distant blocks are scored by `s_b = q·μ_b / √d`, where `μ_b` is the mean of the post-RoPE cached keys in block `b`,
and `q` is the mean post-RoPE query of the query-head group that shares the KV head. The selection is shared across that group.

Nominal supports:

| Support `S` | Distant blocks `r` | Positions per KV head (anchor + window + blocks) |
|---|---|---|
| 97 | 2 | 1 + 64 + 32 |
| 193 | 8 | 1 + 64 + 128 |
| 385 | 20 | 1 + 64 + 320 |
| 769 | 44 | 1 + 64 + 704 |

With reuse `R`, distant blocks are reselected at one-based decode steps `1, 1+R, 1+2R, …`. The paper's sweep uses `R = 1`.

The cache is not truncated. The controller adds a per-head additive mask to each layer's attention during 1-token decode steps:
the anchor, window, and selected blocks are kept, and every other cached key is masked.

The paper also specifies `shared`, `head_union`, `rerank`, and `per_head` as budget-matched alternatives. This repository implements only `kv_group`, the variant used in the survival sweep.

## Experiment: does the principle hold at these supports?

Model: Qwen2-0.5B (float16, SDPA). Contexts: 2K, 8K, 16K tokens of WikiText-2. Supports: `S = 97, 193, 385, 769`.
Five prefixes per context, `R = 1`, 128 greedy tokens. That is 60 paired runs over 15 prefixes.

Both the dense and sparse runs start from the same dense-prefilled cache and the same first token.

- **Survival**: whether the free-running sparse greedy output matches the dense output exactly for 128 tokens. `Tdiv` is the index of the first mismatched token in the 128-token reference (index 0 is the shared first token).
- **Teacher-forced match (TF)**: sparse argmax agreement with the dense tokens, teacher-forced on the dense continuation, excluding the shared first token.
- **ΔPPL**: `100 · (exp(mean sparse NLL) − exp(mean dense NLL)) / exp(mean dense NLL)`, computed on the dense continuation (teacher-forced, not free-running, not held-out perplexity).

### Results

The full per-run table is in [results/qwen_survival_clean.json](results/qwen_survival_clean.json).

- **Survival**: 0 of 60 runs match dense decoding exactly through 128 tokens. The margin condition is sufficient, so none of these runs satisfies it.
- **Distributional quality**: for `S ≥ 193`, 7 of 9 context-support cells have median teacher-forced ΔPPL within 5% of dense.
  Prompt-level ranges include severe 16K outliers: up to +9877% at `S = 97` and +152% at `S = 769`.
- **Warning regime**: all 7 runs with TF below 70% have ΔPPL above +100%. They come from two 16K prefixes (0 and 4).
  This is a descriptive warning regime, not a validated threshold.

### Smoke rerun

`python validate.py --smoke` reruns 2K / prompt 0 / `S = 97, 769` and compares them to the archive.
On the reference machine (RTX 4090 Laptop GPU, 16 GB, torch 2.6.0+cu124, transformers 4.57.1, float16, SDPA), every compared field matches the archived row.

| Support | Tdiv | TF | ΔPPL |
|---|---|---|---|
| 97 | 2 | 93.70% | +5.07% |
| 769 | 22 | 98.43% | −0.34% |

`validate.py` prints `MATCH` or `DIFF` for each field, using a tolerance of 1e-3. Hardware or library differences can change float16 results, so a `DIFF` reports a mismatch against the archive. It does not by itself refute the principle.

## Repository layout

| Path | What it is |
|---|---|
| [condensate/selector.py](condensate/selector.py) | Block budget, block ranges, kv_group selector, reuse schedule, and additive decode mask |
| [condensate/decode.py](condensate/decode.py) | Layer-local sparse decode controller, dense reference, and paired survival metrics |
| [condensate/data.py](condensate/data.py) | WikiText-2 prefix construction |
| [scripts/run_survival.py](scripts/run_survival.py) | Survival sweep CLI; defaults reproduce the paper's 60-run grid |
| [tests/test_selector.py](tests/test_selector.py) | CPU checks: block budget, sharing, reuse cadence, mask, omitted-mass identity |
| [validate.py](validate.py) | Runs the unit tests; with `--smoke`, reruns one paired case and compares it to the archive |
| [results/qwen_survival_clean.json](results/qwen_survival_clean.json) | Archived 60-run sweep behind the paper's survival table |
| [results/rerun_2048_prompt0.json](results/rerun_2048_prompt0.json) | Rerun of the 2K / prompt 0 / S=97,769 case from this repository |

## Reproducing

Requirements: a CUDA GPU with enough memory for Qwen2-0.5B in float16, Python 3.10+, and the packages in [requirements.txt](requirements.txt).

```bash
pip install -r requirements.txt

# CPU checks (selector, mask, omitted-mass identity)
python validate.py

# GPU: rerun one paired case and compare to the archive
python validate.py --smoke

# GPU: the full 60-run sweep (writes JSON after every row)
python scripts/run_survival.py --output results/survival_rerun.json

# Continue a stopped sweep from its saved rows (same settings required)
python scripts/run_survival.py --resume --output results/survival_rerun.json
```

The CLI defaults match the paper: `Qwen/Qwen2-0.5B`, contexts `2048,8192,16384`, supports `97,193,385,769`, `--reuse 1`, `--prompts 5`, `--max-new-tokens 128`.
The model and dataset are downloaded on first use.

## Scope and limits

- One small model (Qwen2-0.5B) and five prefixes per context. The results do not establish a population failure rate or a universal limit on sparse attention.
- The `κ_l` constants are not computed, so the margin condition is not checked by this code.
- Only the kv_group selector is implemented. The paper's other budget-matched variants (`shared`, `head_union`, `rerank`, `per_head`) are not included here.
- This repository makes no timing claim.

## Citation

See [CITATION.cff](CITATION.cff).

## License

Released under the [MIT License](LICENSE).
