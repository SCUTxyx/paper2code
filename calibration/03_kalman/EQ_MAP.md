# 公式—代码对照表:Kalman Filter

| 论文位置 | 公式 | 实现位置 |
|---|---|---|
| 预测 | $x_t^- = F x_{t-1}$ | impl/core_eq.py:L29 |
| 预测 | $P_t^- = F P_{t-1} F^\top + Q$ | impl/core_eq.py:L30 |
| 增益 | $K_t = P_t^- H^\top (H P_t^- H^\top + R)^{-1}$ | impl/core_eq.py:L33(S 在 L32) |
| 更新 | $x_t = x_t^- + K_t(y_t - H x_t^-)$ | impl/core_eq.py:L35 |
| 更新 | $P_t = (I-K_tH)P_t^-$(Joseph 形式,代数恒等) | impl/core_eq.py:L40 |
| 等价表述(信息形式) | $\Lambda_t = P_0^{-1}+\sum H^\top R^{-1}H$,$\eta_t$ 同理 | impl/core_pseudo.py:L45-47 |
| 等价表述(信息形式) | $x_t = \Lambda_t^{-1}\eta_t$ | impl/core_pseudo.py:L48 |
| 后验增益恒等式 | $K_t = \Lambda_t^{-1}H^\top R^{-1}$ | impl/core_pseudo.py:L55 |
