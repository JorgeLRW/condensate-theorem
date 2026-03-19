# Known Solutions Manual

*Sections §1–§92 document earlier experimental findings (Q1–Q19) in the geometric MLP / grokking research arc and are maintained in the researcher's private working notes. This file captures §93–§96 (Q20–Q23), the first sections committed to the repository. Future sections will be appended here.*

---

## §93 — Q20: Geometry Follows Learning, Not the Reverse

**Question:** Does imposing geometric structure on a network accelerate or improve learning?

**Setup:** Compared networks with pre-imposed geometric constraints against unconstrained networks trained on modular arithmetic (grokking) tasks.

**Finding:** Geometric structure *emerges* from training; it cannot be profitably imposed beforehand.

- Networks with hard-coded geometric priors did not converge faster or to better solutions.
- Unconstrained networks developed their own internal geometry spontaneously during grokking.
- The geometry that forms is task-specific and arises as a consequence of the learned solution, not a prerequisite for it.

**Conclusion:** The causal arrow points from learning → geometry, not geometry → learning. Architectural inductive biases that assume a fixed geometric form are fighting the optimizer.

---

## §94 — Q21: Depth Needs Adequate Width

**Question:** Does increasing depth unconditionally improve performance on grokking tasks?

**Setup:** Swept over (depth, d_model) combinations. Held parameter count roughly constant while varying the depth-vs-width trade-off.

**Finding:** Deeper networks without sufficient width fail to generalize.

| Depth | d_model | Generalization |
|-------|---------|----------------|
| 2     | 128     | ✅ Grokks      |
| 4     | 128     | ✅ Grokks      |
| 6     | 128     | ⚠️  Slower     |
| 8     | 128     | ❌ Fails       |
| 8     | 256     | ✅ Grokks      |

- There is a critical depth-to-width ratio below which grokking breaks.
- Width (d_model) determines the representational capacity available per layer; depth amplifies the requirement.
- Simply stacking layers without scaling width creates an information bottleneck that prevents the emergence of the geometric solution.

**Conclusion:** Depth is not free. For every doubling of depth, width must be scaled proportionally to maintain generalization.

---

## §95 — Q22: SwiGLU Dominates Because Gating = Implicit Geometric Awareness

**Question:** Why does SwiGLU consistently outperform ReLU, GELU, and other activations on grokking tasks?

**Setup:** Systematic ablation of activation functions across depth/width combinations. Measured final validation accuracy and epoch-to-grokk.

**Key result:** SwiGLU dominated at every depth tested (2–8 layers).

| Activation | Avg. Grokking Epoch | Final Val Acc |
|------------|---------------------|---------------|
| ReLU       | 4,200               | 91.3%         |
| GELU       | 3,800               | 93.1%         |
| SiLU       | 3,400               | 94.7%         |
| SwiGLU     | **2,100**           | **98.6%**     |

**Interpretation:** The gating component of SwiGLU (`x * σ(Wx)`) acts as an *implicit geometric router*. Rather than applying a fixed nonlinearity, the gate learns to selectively amplify or suppress activations based on their geometric relevance to the current task.

- This is functionally equivalent to the network learning *which dimensions of its representation matter* at each forward pass.
- The gate provides soft geometric awareness without requiring explicit geometric supervision.
- As depth increases, the advantage of SwiGLU over un-gated activations widens, consistent with the gate accumulating richer geometric context across layers.

**Conclusion:** SwiGLU's superiority is not merely empirical. The gating mechanism is a form of learned geometric routing, and grokking tasks reward precisely this capability.

---

## §96 — Q23: You Cannot Inject Awareness, But You CAN Guide Discovery

**Question:** Can we accelerate or improve grokking by explicitly injecting geometric information (e.g., Fourier features, distance matrices, pre-computed embeddings)?

**Setup:** Tested three injection strategies against a feedback-based alternative:
1. **Fourier injection** — prepend Fourier features of input tokens.
2. **Distance injection** — augment embeddings with pairwise modular distances.
3. **Polar injection** — project inputs onto a polar coordinate basis (coefficient 0.1).
4. **Feedback MLP** — an auxiliary head whose output is fed back as an additional input signal in the next forward pass (no hard-coded geometry).

**Results:**

| Method              | Grokking Epoch | Final Val Acc | Notes                          |
|---------------------|----------------|---------------|--------------------------------|
| Baseline (SwiGLU)   | 2,100          | 98.6%         | —                              |
| Fourier injection   | 2,050          | 98.4%         | Marginal, not significant      |
| Distance injection  | 1,980          | 98.0%         | Slightly faster, lower ceiling |
| Polar injection     | ❌ Diverged    | —             | Violent reconciliation at 0.1  |
| **Feedback MLP**    | **1,750**      | **99.1%**     | Tied/beat SwiGLU at all depths |

Depth × feedback-advantage correlation: **ρ ≈ 0.70–0.95** (stronger effect at greater depth).

**Interpretation:**

- Explicit injection forces the network to reconcile a pre-supposed geometry with the geometry it would naturally discover. When these conflict (as they often do), training destabilizes.
- The polar experiment was the clearest failure: a coefficient of 0.1 was too strong, causing oscillatory reconciliation. The geometry imposed did not match the geometry the optimizer wanted.
- Feedback avoids this conflict entirely. Instead of *telling* the network what the geometry is, the feedback signal lets the network *report* on its own intermediate geometric state and act on that report in the next pass.
- The depth × advantage interaction (ρ ≈ 0.70–0.95) is the key result: deeper networks benefit *more* from feedback. This aligns with §93–§95 — deeper networks develop richer geometry and can therefore extract more signal from a feedback channel.

**Conclusion:** You cannot inject geometric awareness from outside. You *can* create a channel through which the network's own emerging awareness feeds back on itself. Feedback beats injection, and the advantage grows with depth.

---

## Future Direction — Q24 Candidate: Feedback Scaling Law at LM Scale

**Background:** The Q20–Q23 arc has established:

| # | Finding |
|---|---------|
| §93 | Geometry follows learning, not the reverse |
| §94 | Depth needs adequate width |
| §95 | SwiGLU dominates because gating = implicit geometric awareness |
| §96 | Feedback beats injection; advantage grows with depth |

**Open question:** All feedback experiments were conducted at `d_model=128` on toy grokking tasks (modular arithmetic). The depth × awareness interaction (ρ ≈ 0.70–0.95) predicts the advantage *grows* with scale. But does it hold on a real language modeling task?

**Proposed experiment — Q24:**

> Test feedback MLP vs. SwiGLU baseline on a Pythia-scale language modeling setup (≈ 2.66 M non-embedding parameters, WikiText-103) at 2–3 model sizes.

| Model size | d_model | Depth | Parameters (NE) |
|------------|---------|-------|-----------------|
| Small      | 128     | 6     | ~0.5 M          |
| Medium     | 256     | 8     | ~2.66 M         |
| Large      | 512     | 10    | ~10 M           |

**Decision criteria:**

- If feedback's perplexity advantage persists at LM scale → publishable architectural contribution (a SwiGLU variant that develops its own geometric routing through gradient discovery).
- If the advantage does not transfer → defines the boundary of the grokking→LM generalization and identifies where geometric self-awareness through feedback breaks down.

**Alternative direction (lower priority):** Revisit polar injection with a smaller coefficient (0.01 instead of 0.1). The reconciliation at 0.1 was violent; a gentler coefficient might converge without oscillation. However, this is a hyperparameter search, not a conceptual advance.

**Recommendation:** Q24 = feedback MLP at LM scale. It is the sharpest available test of whether geometric self-awareness through gradient discovery transfers beyond toy grokking, and the upside (a new publishable activation/routing primitive) justifies the compute cost.
