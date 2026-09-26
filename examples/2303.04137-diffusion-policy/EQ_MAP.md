# Formula-to-code map: Diffusion Policy math core

| Paper location | Formula | Implementation |
|---|---|---|
| §3 (Ho et al. Eq.4) | $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ | impl/core_eq.py:L25-26 (`q_sample`) |
| §3 (Ho et al. Eq.2) | $x_s = \sqrt{1-\beta_s}x_{s-1} + \sqrt{\beta_s}\epsilon_s$ | impl/core_pseudo.py:L12-18 |
| §3 Eq.(5) | $L = \mathrm{mean}(\epsilon-\hat\epsilon)^2$ | impl/core_eq.py:L64-68 (`training_loss`) |
| Song et al. Eq.(12), η=0 | $x_{\tau-1} = \sqrt{\bar\alpha_{\tau-1}}\,\hat x_0 + \sqrt{1-\bar\alpha_{\tau-1}}\,\hat\epsilon$ | impl/core_eq.py:L50-51 (`ddim_step`) |
| Song et al. Eq.(12) | $\hat x_0 = (x_\tau - \sqrt{1-\bar\alpha_\tau}\hat\epsilon)/\sqrt{\bar\alpha_\tau}$ | impl/core_eq.py:L50 |
| §"receding horizon control" | replan every $T_a$, execute chunk head; final chunk to $\min(T_p, H-t)$ | impl/core_eq.py:L73-100 (`execute_receding`) |
| §"receding horizon control" (loop form) | explicit-clock executor | impl/core_pseudo.py:L33-47 |
