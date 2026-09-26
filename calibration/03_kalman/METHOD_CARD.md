# 方法卡片:Kalman Filter(Kalman 1960)

- 论文:*A New Approach to Linear Filtering and Prediction Problems*(R. E. Kalman, 1960, ASME J. Basic Engineering)
- 范围声明:只覆盖离散线性高斯模型的滤波递推(预测 + 更新);连续时间、预测平滑(RTS)、非线性扩展(EKF/UKF)不覆盖。

## 问题形式化

线性高斯状态空间模型:
$$x_t = F x_{t-1} + w_t,\quad w_t\sim\mathcal N(0,Q);\qquad y_t = H x_t + v_t,\quad v_t\sim\mathcal N(0,R)$$
目标:递推计算后验 $\mathbb E[x_t\mid y_{1:t}]$ 与 $\mathrm{Cov}[x_t\mid y_{1:t}]$。

## 符号表

| 符号 | 含义 | 出处 |
|---|---|---|
| $x_t, y_t$ | 状态 / 观测 | §滤波方程 |
| $F, H$ | 状态转移 / 观测矩阵 | 同上 |
| $Q, R$ | 过程 / 观测噪声协方差 | 同上 |
| $P_t$ | 后验协方差 | 同上 |
| $K_t$ | Kalman 增益 | gain equation |

## 核心公式(滤波方程组)

- 预测:$x_t^- = F x_{t-1}$,$P_t^- = F P_{t-1} F^\top + Q$
- 增益:$K_t = P_t^- H^\top (H P_t^- H^\top + R)^{-1}$
- 更新:$x_t = x_t^- + K_t(y_t - H x_t^-)$,$P_t = (I - K_t H)P_t^-$

**等价表述(信息形式,双实现 B 的依据,手写推导)**:静态模型($F=I,Q=0$)下,信息矩阵 $\Lambda_t = P_t^{-1}$ 与信息向量 $\eta_t = \Lambda_t x_t$ 满足
$$\Lambda_t = P_0^{-1} + \sum_{s\le t} H^\top R^{-1} H,\qquad \eta_t = P_0^{-1}x_0 + \sum_{s\le t} H^\top R^{-1} y_s,\qquad x_t = \Lambda_t^{-1}\eta_t$$
(对高斯后验 $\Sigma\propto(\Lambda)^{-1}$ 的精度加权直接展开;动态模型时 $\Lambda_t = (F\Lambda_{t-1}^{-1}F^\top+Q)^{-1} + H^\top R^{-1}H$。)

## claims 清单

- **K1**(高斯后验的充分统计量):静态模型 + 扩散先验($P_0\to\infty$)时,递推后验均值 = 批量最小二乘解 `np.linalg.lstsq`(校准点,零额外依赖真值)。
- **K2**(同上,带信息先验):后验均值 = GLS 闭式 $(H^\top R^{-1}H + P_0^{-1})^{-1}(H^\top R^{-1}y_{1:T} + P_0^{-1}x_0)$。
- **K3**(信息累加):静态模型下后验协方差按 Loewner 序单调不增 → trace 单调不增。
- **K4**(高斯后验对观测线性):$x_T$ 对全部观测的雅可比 = $\Lambda_T^{-1}H^\top R^{-1}$ 按观测块平铺。
- **K5**(新息一致性,估计理论标准结论):模型正确设定时,归一化新息平方 NIS $=\nu^\top S^{-1}\nu$ 的期望 = 观测维数 $d_y$。
