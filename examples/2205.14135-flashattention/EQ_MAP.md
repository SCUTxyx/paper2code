# Formula-to-code map: FlashAttention tiled softmax

| Paper location | Formula | Implementation |
|---|---|---|
| Vaswani 2017 Eq.(1) (the target) | $\mathrm{softmax}(QK^\top/\sqrt d)V$ | impl/core_eq.py:L26 (`attention_direct`) |
| Algorithm 1 | score block $S = Q_i K_j^\top/\sqrt d$ | impl/core_eq.py:L60 |
| Algorithm 1 | $m^{\rm new} = \max(m^{\rm old}, \mathrm{rowmax}(S))$ | impl/core_eq.py:L63 |
| Algorithm 1 | $\tilde P = \exp(S - m^{\rm new})$ | impl/core_eq.py:L64 |
| Algorithm 1 | rescale factor $e^{m^{\rm old}-m^{\rm new}}$ | impl/core_eq.py:L65 |
| Algorithm 1 | $\ell^{\rm new} = e^{\Delta m}\ell^{\rm old} + \mathrm{rowsum}(\tilde P)$ | impl/core_eq.py:L66 |
| Algorithm 1 (normalized variant) | $\mathrm{diag}(e^{\Delta m}\ell^{\rm old})O^{\rm old} + \tilde P V$ | impl/core_eq.py:L67 |
| Algorithm 1 (normalized variant) | $\mathrm{diag}(\ell^{\rm new})^{-1}\cdot(\cdot)$ | impl/core_eq.py:L68 |
| §3.2 | causal mask applied per tile | impl/core_eq.py:L61-62 |
| Algorithm 1 (printed, streaming form) | $U^{\rm new} = e^{\Delta m}U^{\rm old} + \tilde P V$ (unnormalized) | impl/core_pseudo.py:L43 |
| Algorithm 1 (printed, streaming form) | final $O = U/\ell$, once | impl/core_pseudo.py:L46 |
| §3.1 stability device | naive exp for the overflow documentation only | impl/core_pseudo.py:L58 |
