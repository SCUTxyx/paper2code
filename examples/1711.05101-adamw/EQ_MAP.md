# Formula-to-code map: AdamW

| Paper location | Formula | Implementation |
|---|---|---|
| Algorithm 2 | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ | impl/core_eq.py:L24 |
| Algorithm 2 | $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ | impl/core_eq.py:L25 |
| Algorithm 2 | $\hat m_t = m_t/(1-\beta_1^t)$ | impl/core_eq.py:L26 |
| Algorithm 2 | $\hat v_t = v_t/(1-\beta_2^t)$ | impl/core_eq.py:L27 |
| Algorithm 2 (η_t=1) | $\theta_t = \theta_{t-1} - \mathrm{lr}(\hat m_t/(\sqrt{\hat v_t}+\varepsilon)) - \mathrm{lr}\,\lambda\,\theta_{t-1}$ | impl/core_eq.py:L29 |
| Claim W5 (composition) | $\theta_t = \mathrm{AdamStep}(\theta_{t-1},g_t) - \mathrm{lr}\,\lambda\,\theta_{t-1}$ | impl/core_pseudo.py:L14-29 (`adam_step` + separate shrink) |
