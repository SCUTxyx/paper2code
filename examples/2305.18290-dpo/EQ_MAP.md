# Formula-to-code map: DPO

| Paper location | Formula | Implementation |
|---|---|---|
| Eq.(7) | $z = \beta\big[(\log\pi_w - \log\pi_{\mathrm{ref},w}) - (\log\pi_l - \log\pi_{\mathrm{ref},l})\big]$ | impl/core_eq.py:L29 |
| Eq.(7) | $L = -\log\sigma(z) = \mathrm{softplus}(-z)$ | impl/core_eq.py:L30 |
| Eq.(6) | $p = \sigma(z)$ (Bradley–Terry, independent formulation) | impl/core_pseudo.py:L21 |
| Eq.(7) gradient | $\partial L/\partial z = -\sigma(-z)$, four-path chain rule | impl/core_eq.py:L41-46 |
| Eq.(4) | $\pi^*(y\|x) = \pi_{\mathrm{ref}}e^{r/\beta}/Z(x)$ | impl/core_pseudo.py:L31-34 |
| Eq.(5) | $r = \beta\log\frac{\pi}{\pi_{\mathrm{ref}}} + \beta\log Z(x)$ | impl/core_pseudo.py:L41 |
