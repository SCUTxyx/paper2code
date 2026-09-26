# Testability plan: RoPE

| Claim | Content | Test type | Location | Notes |
|---|---|---|---|---|
| R1 | Rotation orthogonal: norm preservation, $R^\top R=I$ | property-invariance | test_properties.py::test_rotation_preserves_norm | 1e-12 |
| R2 | Relative-position invariance (core claim) | property-invariance | test_properties.py::test_relative_position_invariance | several (m,n,c), 1e-12 |
| R3 | Complex form = matrix form | cross-check | test_crosscheck.py | dual formulations |
| R4 | Composition law $R^m R^n = R^{m+n}$ | property | test_properties.py::test_composition | 1e-12 |
| R5 | Gradient $R^{m\top}w$ | gradient check | test_gradients.py | includes score w.r.t. q |
| — | d=2 hand case: $f([1,0],3) = [\cos 3, \sin 3]$;$\langle f(q,3),f(k,1)\rangle = \sin 2$ | anchor | test_anchor.py | standard trig constants, derivation in comments |
| §3.4.3 | Inner-product upper bound decays with relative distance | **not automatable** | → GAP_LIST #1 | asymptotic bound, vector-structure-dependent |
