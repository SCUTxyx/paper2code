---
name: paper2code
description: Reproduce the core method of a paper, with verification. Input an arXiv link or paper file; output a minimal numpy implementation + green gradient/property tests + a method card + a formula-to-code map + an honest gap list. Triggers: reproduce this paper, paper to code, implement the method from this paper, verify the paper's formulas, minimal implementation of this paper's method, 复现这篇论文, 把论文变成代码, 实现论文方法, 验证论文公式, 论文复现.
---

# paper2code: turn "reproduce a paper's core method" into a standard, verified action

**Principle: verify the math, don't run the training.** Input: a paper. Output: a minimal
numpy implementation of the core module + green tests + method card + formula-to-code map +
gap list. Fully automated, zero heavy dependencies (torch/jax/tf are forbidden).

**When NOT to use this skill**: the user wants "run the official repo", "reproduce the
SOTA table", or "end-to-end training" — those are out of scope (missing data / GPU /
training time). Say so honestly, and offer to verify the testable math piece only.

## Artifact contract (fixed for every run; do not add, drop, or rename)

Create `repros/<arxiv-id>-<short-name>/` with the seven-part contract (7 parts = 11 files; "seven" refers to the seven top-level entries in the tree below):

```
METHOD_CARD.md        # method card: problem formalization, symbol table, core formulas
                      #   (quoted with paper numbering), algorithm box, claims list
TEST_PLAN.md          # claim → test-type mapping (auditable)
impl/core_eq.py       # literal implementation of the paper's formulas
impl/core_pseudo.py   # independent formulation (pseudocode / equivalent derivation)
tests/test_gradients.py / test_properties.py / test_anchor.py / test_crosscheck.py
REPORT.md             # verification report: results matrix, findings (incl. paper
                      #   inconsistencies), known limitations
GAP_LIST.md           # honest gap list: for each item, what is missing (data/GPU/time)
EQ_MAP.md             # formula-to-code map: formula number → file:line
```

Skeletons: `references/templates.md`. Ladder details and tolerances:
`references/verification.md`.

## The six-stage pipeline

### S0 Acquire and parse
1. Prefer arXiv HTML via ar5iv (`https://ar5iv.labs.arxiv.org/html/<id>`); read the whole
   page. Fall back to the PDF (page by page, ≤ 20 pages per read) only if HTML fails.
2. Locate the core-contribution section (usually §3 "method" plus dependent appendices).
   **Deep-read only that section**; skim the rest by headings — this is the main token-cost
   control (full-text skims capped at 2).
3. Run the applicability precheck here for systems/dataset papers (see below).
   **Embodied-AI papers** (robot manipulation / locomotion / VLA / policy learning /
   model-based control / sim benchmarks) use the dedicated precheck:
   `references/embodied_precheck.md` — decision tree (Q1 does a math piece exist →
   Q2 five classes of testable math → Q3/Q4 resource dependencies) and produce the
   three-tier feasibility verdict in the METHOD_CARD header:
   ✅ REPRODUCIBLE-MATH / 🟡 PARTIAL / ❌ NOT-REPRODUCIBLE-HERE.

### S1 Method extraction → METHOD_CARD.md
- Symbol table: every symbol, its meaning, and its source equation number;
- Formula list: each formula quoted in LaTeX with the paper's numbering;
- **Claims list**: every property the paper asserts — degeneracy relations, invariances,
  limit behavior, direction/monotonicity statements;
- The algorithm box, line by line.
- Every "the paper claims X" must carry a source (§ / Eq.). Missing claims here means
  missed tests in S2.

### S2 Testability planning → TEST_PLAN.md
Map every claim to a test type (one table):
- analytically checkable → gradient check / cross-check;
- claimed degeneracy → degeneracy test (parameter extremes recover a known method);
- claimed invariance → invariance test (normalization, permutation, shift);
- toy numbers / closed-form mini-examples in the paper → anchor test;
- cannot be decided → GAP_LIST, stating what is missing.
**Applicability precheck**: for systems papers, test only the testable math pieces
(scheduling formulas, cost models) and route the rest to GAP_LIST; for dataset papers, test
only the statistical protocol; for **embodied papers**, follow
`references/embodied_precheck.md` and emit the three-tier feasibility verdict
(✅ REPRODUCIBLE-MATH / 🟡 PARTIAL / ❌ NOT-REPRODUCIBLE-HERE).
Honest boundaries are the core value of this skill.

### S3 Minimal implementation
Hard constraints (violating any one means redo):
1. **numpy-only**: no import of torch/jax/tf/sklearn;
2. synthetic data, explicitly seeded via `np.random.default_rng(seed)`;
3. **quote the formula before implementing it**: docstrings open with the formula number,
   inline comments carry formula numbers (keeps EQ_MAP generatable);
4. float64;
5. when two independent formulations exist (formula vs pseudocode/equivalent derivation),
   implement both as `core_eq.py` + `core_pseudo.py`; **they must not import each other**
   and must use different computation paths (for cross-checking; defense per
   verification.md §3).

### S4 Verification
- The four-level ladder (verification.md §1): L1 reference cross-check → L2 property tests
  → L3 paper anchors → L4 EQ_MAP.
- All four test kinds are mandatory; gradient checks go through `scripts/gradcheck.py`
  (float64 central differences, step cbrt(eps)·max(1,|x|), rtol<1e-6); statistical
  tolerances must be derived analytically (5σ), never eyeballed.
- Run: `python -m pytest repros/<dir>/tests -q`. **A failing test must never be reported as
  "verified"; failures go into REPORT verbatim — a failure is itself a finding**
  (implementation bug, paper typo, or an overreaching claim).

### S5 Report generation
- REPORT.md: results matrix (cases/passes/max error per file) + findings (incl. paper
  inconsistencies) + known limitations;
- GAP_LIST.md: each item states what is missing (data / GPU / training time / information)
  and its impact;
- EQ_MAP.md: formula number → file:line (grep the line numbers after the code is final;
  never estimate by hand);
- Walk the user through the repro directory and point them at EQ_MAP for human review —
  **final judgment always rests with a human**.

## Reuse and calibration

- Shared tool: `scripts/gradcheck.py` (used by every repro);
- Exams: `calibration/` holds five papers with known ground truth (Adam / Attention /
  Kalman / DDPM / InfoNCE); they double as living templates of the artifact contract;
- **After any change to this SKILL.md or scripts/, run `bash scripts/run_calibration.sh`;
  5/5 green or the change is invalid** (results are appended to CALIBRATION_LOG.md).

## Honesty clauses

1. The artifact structure is fixed — the fixedness is the reliability;
2. Claims that cannot be decided go to GAP_LIST; never silently drop them;
3. The paper's own typos/inconsistencies go into REPORT findings — this is what separates
   this skill from an "implementation machine";
4. Every statistical tolerance traces back to an analytic derivation;
5. The report is a verification report, not a certificate of correctness.
