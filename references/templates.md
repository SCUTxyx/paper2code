# Artifact templates (the fixed seven-part contract (7 parts, 11 files))

Every run produces the same fixed structure — this fixedness is the source of the skill's
reliability. Directory name: `repros/<arxiv-id>-<short-name>/` (e.g. `2104.09864-rope/`).

```
repros/<arxiv-id>-<short-name>/
├── METHOD_CARD.md        # ① method card
├── TEST_PLAN.md          # ② testability plan
├── impl/
│   ├── core_eq.py        # ③ literal implementation of the paper's formulas
│   └── core_pseudo.py    #    independent formulation (pseudocode / equivalent derivation)
├── tests/
│   ├── test_gradients.py # ④ four test kinds (fixed file names)
│   ├── test_properties.py
│   ├── test_anchor.py
│   └── test_crosscheck.py
├── REPORT.md             # ⑤ verification report
├── GAP_LIST.md           # ⑥ honest gap list
└── EQ_MAP.md             # ⑦ formula-to-code map
```

## ① METHOD_CARD.md skeleton

```markdown
# Method card: <paper title>
- Paper: <title>, arXiv:<id> (<venue year>)
- Scope statement: this card covers only the "core method section"; everything else is
  explicitly declared out of scope.

## Problem formalization
<input / output / objective, two or three sentences>

## Symbol table
| Symbol | Meaning | Source |
|---|---|---|
| x_t | … | Eq.(n) / §m |

## Core formulas (one per line, quoted in LaTeX)
- Eq.(n):$...$ — <one-line explanation>

## Algorithm box
<Algorithm n verbatim, or a faithful transcription>

## Claims list (every property the paper asserts)
- <id> <claim> (source §m / Eq.n)
```

## ② TEST_PLAN.md skeleton

One row per claim, mapped to a test type; what cannot be tested states its destination.

```markdown
| Claim | Content | Test type | Location | Notes |
|---|---|---|---|---|
| C1 | … | property-invariance | test_properties.py::test_x | … |
| C5 | … | not automatable | → GAP_LIST #2 | needs real data |
```

## ③④ Implementation and test conventions

- `core_eq.py`: implement the paper's formulas line by line; every function docstring cites
  its formula number; inline comments carry formula numbers (keeps EQ_MAP generatable).
- `core_pseudo.py`: an **independent formulation** (pseudocode transcription / equivalent
  closed form / loop formulation); it must not import `core_eq`.
- numpy-only, float64; randomness via `np.random.default_rng(seed)`.
- Standard test-file header (isolates same-named impl modules across repros):

```python
import sys
from pathlib import Path
IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]          # guard against cross-repro module-name collisions
import core_eq
```

- Gradient checks always go through `from gradcheck import assert_grad_close`
  (scripts/ is put on the path by the repository conftest.py).

## ⑤ REPORT.md skeleton

```markdown
# Reproduction report: <short-name> (arXiv:<id>)

## Scope statement
<what is and is not covered, and why>

## Results matrix
| Test file | Cases | Pass | Fail | Max rel error / notes |
|---|---|---|---|---|

## Findings (including paper inconsistencies)
1. …

## Known limitations
- …
```

## ⑥ GAP_LIST.md skeleton

Each item states **what is missing** (data / GPU / training time / information) and the impact.

```markdown
| # | Gap | What's missing | Impact |
|---|---|---|---|
| 1 | … | real dataset | headline numbers not verified; math core only |
```

## ⑦ EQ_MAP.md skeleton

```markdown
| Paper location | Formula | Implementation |
|---|---|---|
| Eq.(7) | $...$ | impl/core_eq.py:L<n> |
| Algorithm 1 L3 | $m_t = ...$ | impl/core_eq.py:L<n> |
```

## How to run

- One repro: `python -m pytest repros/<dir>/tests -q`
- Everything (exams + examples): `bash scripts/run_all_tests.sh`
- Skill regression (mandatory after any SKILL.md change): `bash scripts/run_calibration.sh`
