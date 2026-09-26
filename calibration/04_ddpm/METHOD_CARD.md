# 方法卡片:DDPM 前向过程(Ho et al. 2020)

- 论文:*Denoising Diffusion Probabilistic Models*,arXiv:2006.11239(NeurIPS 2020)
- 范围声明:只覆盖 §2、§4 的**前向(扩散)过程**与线性 β 调度;反向去噪网络、训练目标、采样器不覆盖。

## 问题形式化

前向马尔可夫链逐步加噪:
$$q(x_t|x_{t-1}) = \mathcal N(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I),\qquad \alpha_t := 1-\beta_t,\quad \bar\alpha_t := \prod_{s=1}^{t}\alpha_s$$

## 符号表

| 符号 | 含义 | 出处 |
|---|---|---|
| $x_0$ | 干净数据 | §2 |
| $\beta_t$ | 噪声调度(线性:1e-4 → 0.02,T=1000) | §4 |
| $\alpha_t, \bar\alpha_t$ | $1-\beta_t$ 及其累积积 | §2 |
| $\epsilon$ | 标准高斯噪声 | Eq.(4) |

## 核心公式

- Eq.(1):$q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1})$
- Eq.(2):$q(x_t|x_{t-1}) = \mathcal N(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I)$
- Eq.(4)(闭式边缘,核心校准点):$q(x_t|x_0) = \mathcal N(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I)$,采样形式 $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$

**未展开恒等式(手写推导,双实现 B 的依据)**:把 Eq.(2) 递推展开,
$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sum_{s=1}^{t}\sqrt{\tfrac{\bar\alpha_t}{\bar\alpha_s}}\,\sqrt{1-\alpha_s}\;\epsilon_s$$
其中 $\sum_s (\bar\alpha_t/\bar\alpha_s)(1-\alpha_s) = \bar\alpha_t\sum_s(\tfrac{1}{\bar\alpha_s}-\tfrac{1}{\bar\alpha_{s-1}}) = \bar\alpha_t(\tfrac{1}{\bar\alpha_t}-1) = 1-\bar\alpha_t$(望远镜求和,方差与 Eq.(4) 一致)。

## claims 清单

- **D1**(Eq.4):闭式边缘与逐步加噪同分布 —— 蒙特卡洛均值/方差与解析值一致(容差按 5σ/√N 解析导出)。
- **D2**(上面的恒等式):同一噪声序列下「逐步递推」与「未展开加权和」**精确**相等(逐 float 位级一致,容差 1e-10)。
- **D3**(§4 调度声明):$\bar\alpha_t$ 严格单调下降;$\bar\alpha_T \approx 0$(信号几乎完全破坏,$x_T\to\mathcal N(0,I)$)。
- **D4**(Eq.4 线性重参数化):$\partial x_t/\partial x_0 = \sqrt{\bar\alpha_t}I$,$\partial x_t/\partial\epsilon = \sqrt{1-\bar\alpha_t}I$;未展开式对 $\epsilon_s$ 的梯度 = $\sqrt{\bar\alpha_t/\bar\alpha_s}\sqrt{1-\alpha_s}I$。
