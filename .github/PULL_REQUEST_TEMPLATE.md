# Reproduction submission

One PR = one reproduction (or one documentation/tooling change).

## Contract checklist (meta-tests enforce this — CI will check again)

- [ ] `examples/<arxiv-id>-<short-name>/` with METHOD_CARD / TEST_PLAN / EQ_MAP / REPORT / GAP_LIST
- [ ] `impl/core_eq.py` + `impl/core_pseudo.py` — two independent formulations, no cross-imports
- [ ] Four test kinds: gradients / properties / anchor / crosscheck
- [ ] numpy-only, float64, seeded RNG; no heavy dependencies
- [ ] REPORT results matrix matches actual test collection (`tests/test_report_counts.py`)
- [ ] EQ_MAP line references verified (`tests/test_eq_maps.py`)
- [ ] `python -m pytest -q` green, and `bash scripts/run_calibration.sh` reports 5/5

## Findings

Describe what the verification surfaced (implementation traps, paper inconsistencies,
honest gaps). A reproduction without a finding is fine — but a real finding is gold.
