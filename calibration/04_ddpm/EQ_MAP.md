# 公式—代码对照表:DDPM 前向过程

| 论文位置 | 公式 | 实现位置 |
|---|---|---|
| §4 | $\beta_t$ 线性 1e-4→0.02 | impl/core_eq.py:L8-11 |
| §2 记号 | $\bar\alpha_t = \prod_{s\le t}\alpha_s$ | impl/core_eq.py:L15-17 |
| Eq.(4) | $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ | impl/core_eq.py:L22-23 |
| Eq.(2) | $x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon_t$ | impl/core_pseudo.py:L15 |
| 未展开恒等式 | $x_t = \sqrt{\bar\alpha_t}x_0 + \sum_s \sqrt{\bar\alpha_t/\bar\alpha_s}\sqrt{1-\alpha_s}\,\epsilon_s$ | impl/core_pseudo.py:L26-27 |
| Eq.(4) 批量采样 | 蒙特卡洛对照用 | impl/core_pseudo.py:L31-36 |
