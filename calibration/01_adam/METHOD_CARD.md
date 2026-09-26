# 方法卡片:Adam(Kingma & Ba 2015)

- 论文:*Adam: A Method for Stochastic Optimization*,arXiv:1412.6980(ICLR 2015)
- 范围声明:只覆盖 §2 与 Algorithm 1 的更新规则本身;正则化变体(AMSGrad,ICLR 2018 后续)不覆盖。

## 问题形式化

无约束随机优化 $\min_{\theta} f(\theta)$,每步只能拿到随机梯度 $g_t$。目标:步长对梯度尺度不敏感的一阶更新规则。

## 符号表

| 符号 | 含义 | 出处 |
|---|---|---|
| $\theta_t$ | 第 $t$ 步参数 | Algorithm 1 |
| $g_t$ | 随机梯度 $\nabla_\theta f_t(\theta_{t-1})$ | Algorithm 1 |
| $m_t, v_t$ | 梯度一阶/二阶矩的指数滑动平均 | Algorithm 1 |
| $\hat m_t, \hat v_t$ | 偏差修正后的一/二阶矩 | Algorithm 1 |
| $\beta_1, \beta_2$ | 矩衰减率(默认 0.9 / 0.999) | Algorithm 1 |
| $\alpha$ | 步长(学习率) | Algorithm 1 |
| $\varepsilon$ | 数值稳定项(默认 1e-8) | Algorithm 1 |

## 核心公式(Algorithm 1,逐行)

- L7-8:$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$,$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
- L9-10:$\hat m_t = m_t/(1-\beta_1^t)$,$\hat v_t = v_t/(1-\beta_2^t)$
- L11:$\theta_t = \theta_{t-1} - \alpha\,\hat m_t/(\sqrt{\hat v_t} + \varepsilon)$

## claims 清单

- **A1**(Algorithm 1 + §3 注):偏差修正使首步步长 $\approx \alpha$,与梯度尺度无关(首步 $\hat m_1 = g_1$、$\hat v_1 = g_1^2$,更新 $=\alpha\, g_1/(|g_1|+\varepsilon)$)。
- **A2**(§2 滑动平均的极端情形):$\beta_1,\beta_2 \to 0$ 时矩退化为当前梯度本身,更新退化为 $\alpha\, g/(|g|+\varepsilon) \approx \alpha\,\mathrm{sign}(g)$(符号型更新)。
- **A3**(递推展开,手写推导):$m_t = (1-\beta_1)\sum_{k\le t}\beta_1^{\,t-k} g_k$,$v_t = (1-\beta_2)\sum_{k\le t}\beta_2^{\,t-k} g_k^2$ —— 双表述实现的对拍基础。
- **A4**(几何级数):常梯度 $g$ 下对任意 $t$ 精确有 $\hat m_t = g$、$\hat v_t = g^2$(非极限,$\sum\beta^{k} = (1-\beta^t)/(1-\beta)$ 与偏差修正恰好相消)。
