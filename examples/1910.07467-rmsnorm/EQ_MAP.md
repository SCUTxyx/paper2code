# Formula-to-code map: RMSNorm

| Paper location | Formula | Implementation |
|---|---|---|
| §3 Eq.(3)+ε | $\mathrm{RMS}(x) = \sqrt{\tfrac1d\sum_j x_j^2 + \varepsilon}$ | impl/core_eq.py:L18 |
| §3 Eq.(4) | $y = x/\mathrm{RMS}(x) \odot g$ | impl/core_eq.py:L19 |
| C4 backward | $\partial L/\partial x_k = w_kg_k/r - x_k\sum_i w_ig_ix_i/(d\,r^3)$ | impl/core_eq.py:L31 (r at L29) |
| §3 independent form | per-token loop, explicit $\sum x_j^2$ | impl/core_pseudo.py:L10-23 (r at L21) |
