# Formula-to-code map: LoRA

| Paper location | Formula | Implementation |
|---|---|---|
| §4.1 init | $A \sim \mathcal N(0,\sigma^2)$, $B = 0$ | impl/core_eq.py:L12-13 (`init_lora`) |
| §4.1 Eq.(4) | $\Delta W = \frac{\alpha}{r} B A$ | impl/core_eq.py:L25 |
| §4.1 Eq.(4) | $h = W_0 x + \Delta W\, x$ | impl/core_eq.py:L26 |
| L5 | $\partial L/\partial B = \frac{\alpha}{r} w (Ax)^\top$ | impl/core_eq.py:L37 |
| L5 | $\partial L/\partial A = \frac{\alpha}{r} B^\top w\, x^\top$ | impl/core_eq.py:L38 |
| §4.1 (no-ΔW form) | $h = W_0 x + \frac{\alpha}{r} B (A x)$ | impl/core_pseudo.py:L18 |
| §4.4 (deployment) | merged $W = W_0 + \frac{\alpha}{r} B A$ | impl/core_pseudo.py:L24 |
| §4.4 (deployment) | unmerge $W_0 = W - \frac{\alpha}{r} B A$ | impl/core_pseudo.py:L30 |
