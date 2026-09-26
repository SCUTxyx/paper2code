# Contributing to paper2code

Thanks for considering a contribution. The repo has one contribution currency:
**a reproduction** — one paper's seven-part contract (7 parts, 11 files), with green tests. Reports of
paper inconsistencies or failing claims are equally welcome (see below).

## Adding a reproduction (the main path)

1. **Follow the skill, by hand or by agent.** Read [SKILL.md](SKILL.md) and the
   templates in [references/templates.md](references/templates.md). Create
   `examples/<arxiv-id>-<short-name>/` with the full contract:
   `METHOD_CARD.md`, `TEST_PLAN.md`, `impl/core_eq.py` + `impl/core_pseudo.py`
   (two independent formulations — they must not import each other),
   the four test files under `tests/`, `REPORT.md`, `GAP_LIST.md`, `EQ_MAP.md`.
2. **numpy-only, float64, seeded.** No torch/jax/tf/sklearn (a meta-test enforces this).
   All randomness via `np.random.default_rng(seed)`.
3. **Gradient checks** go through `scripts/gradcheck.py` (rtol < 1e-6). Statistical
   tolerances must be derived analytically (5σ) — no eyeballed numbers.
4. **Update your REPORT's results matrix** to match reality. Two meta-tests
   (`tests/test_report_counts.py`, `tests/test_eq_maps.py`) verify that REPORT case
   counts match what pytest collects and that every EQ_MAP line reference points at a
   real, non-blank line.
5. **Run the gates**:
   ```bash
   python -m pytest -q                 # everything, including the meta-tests
   bash scripts/run_calibration.sh     # 5/5 or the change is invalid
   ```

## Reporting paper inconsistencies / failed claims

Open an issue with the paper reference, the claim, and (if possible) a minimal numpy
script demonstrating the discrepancy. Accepted issues become repro folders whose REPORT
documents the finding — these are the repo's most valuable artifacts.

## What we do NOT accept

- Reimplementations without the four test kinds;
- Tolerances that are not analytically derived;
- "Verified" claims with failing tests (failures go in the REPORT, verbatim);
- Datasets, model weights, or heavyweight dependencies.

## Review bar

A reviewer should be able to: run one command, see green; open one EQ_MAP, spot-check
two formulas against the paper in minutes. If your PR cannot offer that, it is not done.
