# Reproduction report: Diffusion Policy math core (arXiv:2303.04137, §3)

## FEASIBILITY verdict

**🟡 PARTIAL** — the action-diffusion math core (DDPM objective, DDIM inference,
receding-horizon execution) is fully verified here; the paper's empirical claims
(real-robot success rates, sim benchmarks, ablations) require weights, hardware and
demonstration data and are honestly out of reach (GAP_LIST).

## Scope statement

Only §3's math: training objective, DDIM sampler (the real-world inference path),
receding-horizon execution semantics. Visual encoder, network ε̂_θ, training dynamics:
out of scope. Per references/embodied_precheck.md.

## Results matrix

| Test file | Cases | Pass | Fail | Notes |
|---|---|---|---|---|
| test_properties.py | 4 | 4 | 0 | receding window (bit-identical under out-of-window mutations); stitching semantics; DDIM biased-prediction deviation (negative property, monotone in bias); perfect-ε exactness |
| test_anchor.py | 3 | 3 | 0 | DDIM trajectory invariance (atol 1e-10, both forms); hand-literal stitching [0,1,10,11,20,21,22]; chunk-head execution |
| test_gradients.py | 2 | 2 | 0 | ∂x_t/∂x0 = √ᾱ_t; ∂L/∂ε̂ = 2(ε̂−ε)/N — both < 1e-6 |
| test_crosscheck.py | 3 | 3 | 0 | closed-form vs iterative same-ε identity (1e-10); direct vs predicted-x_0 DDIM forms (1e-12); two executors (three horizons incl. cyclic-overflow case) |

Run: `python -m pytest examples/2303.04137-diffusion-policy/tests -q` (float64, CPU, < 1 s).

## Findings

1. **Primary-source verification caught a misattribution before it shipped**: the plan
   for this repro assumed Diffusion Policy used exponentially-weighted temporal
   ensembling over overlapping chunks — that mechanism belongs to **ACT**
   (arXiv:2304.13705 §IV-A, `w_i = exp(−m·i)`); Diffusion Policy uses receding-horizon
   *commitment* (execute T_a, replan, optional warm-start). Both ar5iv pages were
   checked (2026-09-26); the confusion is widespread in secondary sources.
2. **The DDIM trajectory-invariance is an exact, cheap, and powerful test**: with
   ε̂ = ε, the closed-form trajectory is an invariant of the DDIM map, so the sampler
   is verified end-to-end without any learned network (atol 1e-10 over 60 steps).
   The negative counterpart (biased ε̂ ⇒ order-1 deviation, monotone in bias) pins why
   predictor quality matters.
3. **Executor semantics needed an explicit convention**: when chunks run out before the
   horizon, the two independent executors initially disagreed (truncate vs cycle the
   final chunk). The contract — "keep executing the final chunk cyclically" — is now
   pinned by a cross-check over three horizons including the overflow case.
4. The DDPM forward-process math is shared with calibration/04_ddpm and re-derived here
   (self-containedness of repro folders); the two suites cross-confirm.

## Known limitations

- Linear β schedule; DP-sim uses squared-cosine with iDDPM (100 steps), real-world uses
  DDIM (10-16 inference steps) — the sampler math is schedule-agnostic but this repro
  exercises the linear schedule only.
- No learned ε̂_θ: all tests use the true ε or simple biases; conditioning and network
  architecture out of scope.
- Warm-starting of the diffusion chain between replans is mentioned as optional by the
  paper and not modeled.
