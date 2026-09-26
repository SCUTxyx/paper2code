# Reproduction report: DPO (arXiv:2305.18290, §4 objective and loss)

## Scope statement

Only the computable pieces of the §4 derivation chain (Eq.3–7): the DPO loss (Eq.7),
the Bradley–Terry preference model (Eq.6), the closed-form optimal policy (Eq.4), and
the reward reparameterization (Eq.5). Training dynamics, the expectation over real
preference data, and comparison against PPO-RLHF are not covered (GAP_LIST).

## Results matrix

| Test file | Cases | Pass | Fail | Notes |
|---|---|---|---|---|
| test_properties.py | 3 | 3 | 0 | P1 log2 degeneracy / P2 monotonicity + swap identity (1e-12) |
| test_anchor.py | 3 | 3 | 0 | z∈{0, ln3, −ln3} → {log2, log(4/3), log4} as exact hand constants |
| test_gradients.py | 2 | 2 | 0 | four-path log-prob gradients < 1e-10; reference-point gradient magnitude β/(2N) exact |
| test_crosscheck.py | 3 | 3 | 0 | Eq.7 literal vs BT path (1e-12); Eq.5 round trip exact; π* normalization |

Run: `python -m pytest examples/2305.18290-dpo/tests -q` (float64, CPU, < 1 s).

## Findings

1. **Use log-prob differences, never probability ratios.** Eq.(7) is written as
   $\beta\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}$ — mathematically identical to a
   log-prob difference — but computing probabilities first and dividing loses precision
   and can underflow (π spans 1e-3 ~ 1e-40). Both writings pass "all properties green"
   easily; their numerical paths differ materially. This is the most common silent bias
   in open-source DPO implementations.
2. **$L(\pi_\theta{=}\pi_{\mathrm{ref}}) = \log 2$ is a free smoke test** for any DPO
   implementation; this reproduction additionally pins the exact reference-point
   gradient magnitude β/(2N).
3. **The $\beta\log Z(x)$ of Eq.(5) cancels in pairwise differences but is required for
   pointwise reward recovery.** The cross-check builds a counterexample (dropping logZ
   fails to recover r), confirming the term's necessity — which explains why DPO
   training needs no partition function while reward inversion does.
4. The swap identity $L(y_l,y_w) - L(y_w,y_l) = z$ holds to 1e-12 and works as a purely
   algebraic (implementation-independent) self-check.

## Known limitations

- The expectation $\mathbb E_{(x,y_w,y_l)}$ is not estimated on a real preference
  distribution (needs a dataset, out of scope).
- No β's empirical effect on KL deviation (needs training), no IPO/KTO comparison.
- The implementation assumes exact log-probs are available (white-box model); black-box
  API models are out of reach (as they are for the paper itself).
