# 公式—代码对照表:DPO

| 论文位置 | 公式 | 实现位置 |
|---|---|---|
| Eq.(7) | $z = \beta\big[(\log\pi_w - \log\pi_{\mathrm{ref},w}) - (\log\pi_l - \log\pi_{\mathrm{ref},l})\big]$ | impl/core_eq.py:L29 |
| Eq.(7) | $L = -\log\sigma(z) = \mathrm{softplus}(-z)$ | impl/core_eq.py:L30 |
| Eq.(6) | $p = \sigma(z)$(Bradley–Terry,独立表述) | impl/core_pseudo.py:L21 |
| Eq.(7) 梯度 | $\partial L/\partial z = -\sigma(-z)$,四路链式 | impl/core_eq.py:L41-46 |
| Eq.(4) | $\pi^*(y\|x) = \pi_{\mathrm{ref}}e^{r/\beta}/Z(x)$ | impl/core_pseudo.py:L31-34 |
| Eq.(5) | $r = \beta\log\frac{\pi}{\pi_{\mathrm{ref}}} + \beta\log Z(x)$ | impl/core_pseudo.py:L41 |
