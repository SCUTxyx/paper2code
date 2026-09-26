# Calibration set: 5 exams with known ground truth

This is the skill's own regression test (PLAN §4.3). After every change to SKILL.md or the
shared scripts, rerun `bash scripts/run_calibration.sh`; results are appended to the root
`CALIBRATION_LOG.md`. **Pass criterion: 5/5 green.** A full run takes seconds.

| Exam | Paper | Calibration point | Truth source (numpy-only) |
|---|---|---|---|
| 01_adam | Adam (Kingma & Ba 2015) | Hand-computed first updates; bias-corrected first step ≈ lr; β→0 degeneracy | Hand-derived reference values + dual implementation |
| 02_attention | Attention (Vaswani 2017) | Softmax rows sum to 1; causal mask zero future dependency; 1/√d scaling | Properties + dual implementation + sigmoid-identity anchor |
| 03_kalman | Kalman Filter (1960) | Static linear-Gaussian: recursion = batch least squares | `np.linalg.lstsq` (zero extra dependencies) |
| 04_ddpm | DDPM (Ho et al. 2020) | Closed-form marginal q(x_t\|x₀) vs iterative noising Monte Carlo | Closed form vs Monte Carlo |
| 05_infonce | InfoNCE / CLIP | Tower-swap symmetry; temperature-limit degeneracy; permutation/scale invariance | Properties + degeneracy + hand-computed anchor |

## Composition of each exam

Each exam mirrors the artifact contract, so the exams double as living templates:
`METHOD_CARD.md`, `TEST_PLAN.md`, `impl/core_eq.py` + `impl/core_pseudo.py`
(dual formulations), the four test kinds under `tests/`, and `EQ_MAP.md`.
The per-exam markdown documents are kept in Chinese as historical artifacts; test code and
comments are in English, and new repros should follow the English templates in
`references/templates.md`.

## Findings recorded while building the exams

- **Kalman (03)**: the textbook covariance update $(I-KH)P^-$ suffers catastrophic
  cancellation at diffuse prior $P_0=10^{10}$ — the posterior covariance turns
  indefinite and the mean drifts ~1e-1 away from batch least squares. The algebraically
  identical Joseph form (1968) yields 2.4e-8 at $P_0=10^8$. A live example of
  "properties all green, numerics still broken" — recorded in
  `03_kalman/tests/test_anchor.py`'s docstring.
- **Attention (02)**: "the loss gradient w.r.t. $V_j$ is zero for future $j$" is a
  **wrong** property statement (later queries do see $V_j$); the correct causal
  property is per-row, $\partial\,\mathrm{out}_i/\partial V_j = 0$ for $j>i$.
  Property statements themselves need verification — which is exactly why TEST_PLAN
  spells out the testing semantics claim by claim.

## Numeric specification

Gradient checks use `scripts/gradcheck.py` (float64 central differences, step
cbrt(eps)·max(1,|x|), rtol < 1e-6); value cross-checks rtol = 1e-5; exact relations
1e-12; statistical tolerances always derived analytically at 5σ.

## Mutation audit (does the verification layer actually have teeth?)

Three classic bugs were injected into committed implementations one at a time
(restored afterwards); the table records which tests killed each mutant — the
suite's *measured* discrimination, not its intention.

| Injected bug | Killed by | Note |
|---|---|---|
| Adam: ε moved inside the sqrt, `m̂/√(v̂+ε)` | 4 tests: cross-check (both configs) + two property tests whose **hand-derived references** encode the correct placement | At the default ε=1e-8 the readings differ by ~1e-8 — ordinary tolerances (rtol 1e-5/1e-6) do NOT see it; the implementation-independent hand references do. A new `ε=1` anchor makes the gap 8-26× and kills it explicitly. |
| LoRA: `(α/r)` → `(α·r)` | 4 tests: the dual-implementation cross-check + new r=2 scale anchor | With r=1 the two readings coincide; discriminators must use r>1. |
| RoPE: θ indexing off-by-one (`i` starts at 1) | 5 tests: all anchors | The three rope formulations share `theta_seq`, so cross-checks are blind by design — only the paper-derived anchors defend here, exactly the division of labor the ladder predicts. |

Conclusions, for the record:

1. **Hand-derived references are implementation-independent oracles** — they caught the
   ε misplacement that value tolerances structurally cannot. Every exam keeps its
   anchors precisely for this.
2. **Cross-checks defend only against asymmetric misreadings** — a misreading shared by
   both formulations is invisible to them (RoPE case) and must be caught by anchors or
   EQ_MAP review. This is why the contract mandates all four test kinds.
3. A tolerance is not a substitute for oracle correctness: an analytically-derived
   tolerance against a *misread* formula is precisely precise about the wrong thing.
