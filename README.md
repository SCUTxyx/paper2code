# paper2code

[![CI](https://github.com/SCUTxyx/paper2code/actions/workflows/ci.yml/badge.svg)](https://github.com/SCUTxyx/paper2code/actions/workflows/ci.yml)

> Turn "reproducing the core method of a paper" into a standard, *verified* action with honest boundaries.
> **Principle: verify the math, don't run the training.**

Give it an arXiv paper; it produces a minimal **numpy implementation** of the core module, a
**green suite of gradient/property tests**, a method card, a formula-to-code map, and an
**honest gap list** of what could *not* be verified. Delivered as an agent skill — fully
automated, zero heavy dependencies.

## Why

| Existing approach | What it gives | What it lacks |
|---|---|---|
| Papers With Code | Index of official/community implementations | No verification, not on-demand |
| labml.ai annotated implementations | Hand-picked annotated code | Sparse coverage, slow updates |
| Just asking an LLM | Fast | No verification, no boundary statement |

paper2code = **verifiable** (properties + gradients + cross-checks + anchors) + **honest
boundaries** (gap lists) + **agent-native** (SKILL.md, fully automated) + **zero heavy
dependencies** (numpy-only).

## Quick start (≤ 5 minutes, no new dependencies)

```bash
git clone https://github.com/SCUTxyx/paper2code.git
cd paper2code
pip install numpy pytest        # skip if already available
python -m pytest -q             # 76 tests: calibration 5/5 + two example repros + tooling
```

To install as an agent skill, drop `SKILL.md` into your skill directory (e.g.
`~/.zcode/skills/paper2code/SKILL.md`), then ask your agent:
"**reproduce this paper: \<arXiv link\>**".

### Do I need a GPU? An API key?

**No, and no.**

- **Compute**: everything in this repo runs on CPU in float64 with small synthetic
  matrices. The whole 76-test suite finishes in ~1.5 s. No GPU, no datasets, no
  model weights, ~100 KB per reproduction.
- **API keys**: the repo is pure local numpy + pytest — nothing phones home. Reading
  papers goes through public arXiv/ar5iv HTML.
- The *only* LLM involvement is your own agent session when it runs the skill on a new
  paper (that costs tokens on your agent platform, not here). The agent's output is then
  verified by CPU-only tests — which is the entire point.

## Verification ladder

| Level | Method | When |
|---|---|---|
| L1 Cross-check | Same input → compare against an authoritative reference (rtol=1e-5) | Strongest, when a reference exists |
| L2 Property tests | Degeneracy / invariances / gradient checks (rtol<1e-6) / limits & monotonicity | The workhorse when no reference exists |
| L3 Paper anchors | Recompute the paper's own toy numbers / closed-form examples | The only chance to check against the authors |
| L4 Formula map | Formula number → file:line | Not automatic; reduces human review to minutes |

Residual risk: **L2 cannot catch a self-consistent misreading** (e.g., ε added in the
wrong place, all properties still green). Defenses = two independent implementations
cross-checked against each other + EQ_MAP for human review. **Final judgment always
rests with a human.**

## Calibration set: the skill's own regression tests (5/5 green)

Five papers with known ground truth; each is run end-to-end, all truth sources numpy-only:

| Exam | Calibration point | Tests |
|---|---|---|
| Adam | Hand-computed first steps; bias-corrected first step ≈ lr; β→0 degeneracy | 10 ✅ |
| Attention | Softmax rows sum to 1; causal mask zero future dependency; 1/√d scaling | 10 ✅ |
| Kalman Filter | Static linear-Gaussian: recursion = batch least squares (`lstsq`) | 7 ✅ |
| DDPM | Closed-form marginal q(x_t\|x₀) vs iterative noising Monte Carlo | 10 ✅ |
| InfoNCE/CLIP | Tower-swap symmetry; temperature limits; permutation/scale invariance | 10 ✅ |

After any change to SKILL.md, rerun `bash scripts/run_calibration.sh`; results are appended
to [CALIBRATION_LOG.md](CALIBRATION_LOG.md). Two genuine findings made while building the
exams (catastrophic cancellation in the diffuse-prior Kalman recursion; a commonly-mistyped
formulation of the causal-attention gradient property) are recorded in
[calibration/README.md](calibration/README.md) — they are themselves evidence the method works.

## Example artifacts (real-paper trial runs)

Every reproduction has a fixed seven-file contract:
`METHOD_CARD / TEST_PLAN / impl (dual implementation) / tests (4 kinds) / REPORT / GAP_LIST / EQ_MAP`.

- **[RoPE rotary position embedding](examples/2104.09864-rope/)** (arXiv:2104.09864) — 13 tests green.
  Finding: the 1-based/0-based indexing convention of θ is the top real-world risk; property
  tests cannot catch an off-by-one that is self-consistent — only the EQ_MAP review can.
  Gap #1: the §3.4.3 decay upper bound is an asymptotic statement and honestly cannot be
  turned into a deterministic test → documented in GAP_LIST.
- **[DPO direct preference optimization](examples/2305.18290-dpo/)** (arXiv:2305.18290) — 11 tests green.
  Findings: use log-prob differences, never probability ratios (numerical stability); the
  β·logZ term in Eq.(5) cancels in pairwise differences but is *required* for pointwise
  reward recovery (verified by a counterexample).

## Cost

| Dimension | Per reproduction | Notes |
|---|---|---|
| Tokens | 50–150K (bounded) | Deep read limited to the method section; full-text skims capped |
| Local CPU | < 10 s | Small-matrix synthetic tests |
| Storage | 30–100 KB / paper | No datasets, no weights, no caches |
| Dependencies | Zero new | numpy / pytest |

## Repository layout

```
SKILL.md              # the skill itself (six-stage pipeline + hard constraints)
references/           # verification ladder details, numeric specs, artifact templates
scripts/              # gradcheck.py (shared finite-difference checker), regression scripts
tests/                # self-tests for gradcheck.py (verify the verifier)
calibration/          # 5 exams (they double as living templates of the artifact contract)
examples/             # real-paper trial runs (RoPE, DPO)
repros/               # local run area for users (.gitignored)
CALIBRATION_LOG.md    # calibration regression log
PLAN.md               # project plan (overview/contract/methodology/verification/milestones/...)
```

## Terms

- Code and reports only — no paper PDFs (arXiv links cited); MIT licensed.
- Contributions welcome via the templates in `references/templates.md`: one PR = one
  paper's seven-file contract + green tests. Papers with suspected typos or failing
  claims are *especially* welcome — those are this repo's most valuable artifacts.

Methodology details: [PLAN.md](PLAN.md) · [SKILL.md](SKILL.md) ·
[references/verification.md](references/verification.md)
