# Formula-to-code map: DDPM forward process

| Paper location | Formula | Implementation |
|---|---|---|
| §4 | $\beta_t$ linear 1e-4→0.02 | impl/core_eq.py:L10-13 (`linear_beta_schedule`) |
| §2 notation | $\bar\alpha_t = \prod_{s\le t}\alpha_s$ | impl/core_eq.py:L16-18 (`alpha_bar_seq`) |
| Eq.(4) | $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ | impl/core_eq.py:L21-25 (`q_sample`) |
| Eq.(2) | $x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon_t$ | impl/core_pseudo.py:L12-19 (`q_sample_iterative`, update at L18) |
| Unrolled identity | $x_t = \sqrt{\bar\alpha_t}x_0 + \sum_s \sqrt{\bar\alpha_t/\bar\alpha_s}\sqrt{1-\alpha_s}\,\epsilon_s$ | impl/core_pseudo.py:L22-31 (`q_sample_unrolled`, sum at L27-30) |
| Eq.(4) batch sampling | Monte Carlo comparison | impl/core_pseudo.py:L34-40 (`q_sample_batch`) |
| Eq.(2) batch sampling | Monte Carlo comparison | impl/core_pseudo.py:L43-49 (`q_sample_iterative_batch`) |
