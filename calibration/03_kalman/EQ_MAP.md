# Formula-to-code map: Kalman Filter

| Paper location | Formula | Implementation |
|---|---|---|
| Predict | $x_t^- = F x_{t-1}$ | impl/core_eq.py:L29 |
| Predict | $P_t^- = F P_{t-1} F^\top + Q$ | impl/core_eq.py:L30 |
| Gain | $K_t = P_t^- H^\top (H P_t^- H^\top + R)^{-1}$ | impl/core_eq.py:L33 (S at L32) |
| Update | $x_t = x_t^- + K_t(y_t - H x_t^-)$ | impl/core_eq.py:L35 |
| Update | $P_t = (I-K_tH)P_t^-$ (Joseph form, algebraically identical) | impl/core_eq.py:L40 |
| Information form | predict $P^- = F\Lambda^{-1}F^\top + Q$ | impl/core_pseudo.py:L38-41 |
| Information form | $\Lambda_t = \Lambda_{t-1} + H^\top R^{-1} H$, $\eta_t += H^\top R^{-1}y$ | impl/core_pseudo.py:L43,45 |
| Information form | $x_t = \Lambda_t^{-1}\eta_t$ | impl/core_pseudo.py:L46 |
| Posterior gain identity | $K_t = \Lambda_t^{-1}H^\top R^{-1}$ | impl/core_pseudo.py:L52 |
