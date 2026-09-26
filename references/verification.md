# Verification ladder and numeric specifications

This document details SKILL.md stage S4. Each reproduction climbs the four-level ladder
from strongest to weakest evidence; always use the highest level available.

## 1. The four-level ladder

| Level | Method | Where the oracle comes from | Pass criterion | Limitation |
|---|---|---|---|---|
| **L1 Cross-check** | Same input → compare against an authoritative reference | scipy / official repos / independent implementations | `rtol=1e-5` | The reference itself may be wrong; strongest when it exists, unavailable otherwise |
| **L2 Property tests** | Degeneracy / invariance / gradient checks / limits & monotonicity | The paper's own mathematical claims | gradients `rtol<1e-6`; exact discrete properties `1e-12` | Cannot catch *self-consistent* misreadings (see §3) |
| **L3 Paper anchors** | Recompute the paper's toy numbers / closed-form mini-examples | The authors themselves | value checks `rtol=1e-5`; statistical checks per §2.3 | Not every paper has reproducible anchors; use them whenever present |
| **L4 Formula map** | EQ_MAP: formula number → file:line | Human | Not automatic | Reduces human review to minutes |

### The four standard test kinds (fixed names in the artifact contract)

- `test_gradients.py` — finite-difference gradient checks: for every differentiable scalar
  output, analytic gradient vs central differences from `scripts/gradcheck.py`.
- `test_properties.py` — degeneracy (parameter extremes recover a known method),
  invariances (normalization, permutation, shift), limits and monotonicity.
- `test_anchor.py` — numeric anchors from the paper; when absent, use hand-verifiable
  closed-form mini-cases (derivation in test comments, constants as literals).
- `test_crosscheck.py` — two independent formulations cross-checked
  (formula version `core_eq.py` vs pseudocode/equivalent-derivation version `core_pseudo.py`).

## 2. Numeric specifications

1. Everything float64; all randomness via explicitly seeded `np.random.default_rng(seed)`.
2. Gradient checks: float64 central differences, relative error **< 1e-6**; step size at the
   central-difference optimum `cbrt(eps)·max(1,|x|)` (≈6e-6, built into `scripts/gradcheck.py`).
3. Value cross-checks (dual implementations / references / closed forms): **rtol = 1e-5**;
   mathematically exact relations (softmax row sums, orthogonal rotations) use **1e-12**.
4. Statistical tests (Monte Carlo comparisons): **tolerances must be derived analytically**
   (e.g. mean tolerance = 5·σ/√N, variance tolerance = 5·σ²·√(2/N)); no eyeballed tolerances.
   Put the derivation in the test comments.
5. Numerical-stability devices (`-inf` masks, log-sum-exp) are implementation details that do
   not change the math; if they affect tolerances, say so in REPORT's limitations.
6. ε-stabilized norms (RMSNorm/LayerNorm-style): the loss's third derivative grows as
   1/r³ with the RMS scale r, so central-difference truncation (h²·f'''/6) explodes at
   small input scales — an artifact of the CHECKER, not the code. When testing at small
   scales, either pick a scale where truncation stays under rtol (and document the
   estimate) or loosen the tolerance with the derivation in the test docstring.

## 3. Residual risk and defenses

**L2 cannot catch a self-consistent misreading** (ε added in the wrong place, a missing
log, properties still all green). Defenses, in order:

1. Two independent formulations cross-checked — a literal transcription of the formulas vs
   a pseudocode / equivalent-derivation version; different computation paths make a shared
   misreading much less likely;
2. EQ_MAP human review — every formula's implementing line listed; a human scan takes minutes;
3. **Final judgment always rests with a human**: REPORT is a verification report, not a
   certificate of correctness.

## 4. Applicability precheck (run in stage S2)

- **Systems papers** (training frameworks, inference systems, distributed schemes): the core
  contribution is not unit-testable. Say so; test only the testable math pieces (schedules,
  cost models), put the rest in GAP_LIST honestly.
- **Dataset/benchmark papers**: the math piece may be just a statistical protocol. Test the
  protocol's consistency; do not recompute benchmark numbers (out of scope).
- **Pure theory papers** (no algorithm box): produce the EQ_MAP only, plus property tests for
  any mechanically checkable lemma; if none, say so honestly.

## 5. Failure handling

- A failing test must **never** be reported as "verified". Failures go into REPORT's results
  matrix and findings verbatim — a failure is itself a finding (implementation bug, paper
  typo, or an overreaching claim).
- Paper inconsistencies (formulas that disagree, undefined symbols, appendix vs main text)
  go into REPORT's findings, with their blast radius noted in GAP_LIST.
