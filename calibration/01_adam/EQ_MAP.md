# Formula-to-code map: Adam

| Paper location | Formula | Implementation |
|---|---|---|
| Algorithm 1 L1 | $m_0 = 0,\ v_0 = 0$ | impl/core_eq.py:L18-19 |
| Algorithm 1 L7-8 | $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ | impl/core_eq.py:L23 |
| Algorithm 1 L7-8 | $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ | impl/core_eq.py:L24 |
| Algorithm 1 L9 | $\hat m_t = m_t/(1-\beta_1^t)$ | impl/core_eq.py:L25 |
| Algorithm 1 L10 | $\hat v_t = v_t/(1-\beta_2^t)$ | impl/core_eq.py:L26 |
| Algorithm 1 L11 | $\theta_t = \theta_{t-1} - \alpha\,\hat m_t/(\sqrt{\hat v_t}+\varepsilon)$ | impl/core_eq.py:L27 |
| claim A3 (closed form) | $m_t = (1-\beta_1)\sum\beta_1^{t-k} g_k$ | impl/core_pseudo.py:L20-32 (`_ema_history`, weighted accumulation at L30-32) |
| claim A3 (closed form) | $v_t = (1-\beta_2)\sum\beta_2^{t-k} g_k^2$ | impl/core_pseudo.py:L43-44 (same helper on $g^2$) |
