# Formula-to-code map: DPO

| Paper location | Formula | Implementation |
|---|---|---|
| Eq.(7) | $z = \beta\big[(\log\pi_w - \log\pi_{\mathrm{ref},w}) - (\log\pi_l - \log\pi_{\mathrm{ref},l})\big]$ | impl/core_eq.py:L31 |
| Eq.(7) | $L = -\log\sigma(z) = \mathrm{softplus}(-z)$ | impl/core_eq.py:L32 |
| Eq.(6) | $p = \sigma(z)$ (Bradley–Terry, independent formulation) | impl/core_pseudo.py:L23 |
| Eq.(7) gradient | $\partial L/\partial z = -\sigma(-z)$, four-path chain rule | impl/core_eq.py:L45-49 |
| Eq.(4) | $\pi^*(y\|x) = \pi_{\mathrm{ref}}e^{r/\beta}/Z(x)$ | impl/core_pseudo.py:L34-37 |
| Eq.(5) | $r = \beta\log\frac{\pi}{\pi_{\mathrm{ref}}} + \beta\log Z(x)$ | impl/core_pseudo.py:L44 |
