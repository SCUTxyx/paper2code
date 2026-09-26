# 方法卡片:DPO 直接偏好优化(Rafailov et al. 2023)

- 论文:*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*,arXiv:2305.18290(NeurIPS 2023)
- 范围声明:只覆盖 §4 的目标推导(Eq.3–7)与损失函数本身;训练动态、RM 阶段、采样策略不覆盖。

## 问题形式化

从偏好对 $(y_w \succ y_l \mid x)$ 学习策略,绕过显式奖励建模:
$$\max_\pi\ \mathbb E_{x\sim\mathcal D,\,y\sim\pi}\big[r(x,y)\big] - \beta\,\mathbb{KL}\big(\pi \| \pi_{\mathrm{ref}}\big)\quad(\text{Eq.3})$$

## 符号表

| 符号 | 含义 | 出处 |
|---|---|---|
| $\pi_\theta, \pi_{\mathrm{ref}}$ | 策略 / 冻结参考策略 | §4 |
| $\beta$ | KL 强度(反比于偏离自由度) | Eq.(3) |
| $y_w, y_l$ | 偏好对中被选 / 被拒的回答 | §4 |
| $z$ | Eq.(7) 中 $\sigma$ 的自变量(隐含奖励差) | Eq.(6)(7) |
| $Z(x)$ | 配分函数(成对差分中消去) | Eq.(4)(5) |

## 核心公式(§4 推导链)

- Eq.(4):KL 约束奖励最大化的闭式最优策略 $\pi^*(y|x) = \frac{1}{Z(x)}\pi_{\mathrm{ref}}(y|x)\exp\big(\frac{1}{\beta}r(x,y)\big)$
- Eq.(5)(奖励重参数化):$r(x,y) = \beta\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta\log Z(x)$
- Eq.(6):Bradley–Terry 偏好模型 $p(y_w \succ y_l) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$
- Eq.(7)(DPO 损失):$L_{\mathrm{DPO}} = -\mathbb E_{(x,y_w,y_l)}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\Big)\Big]$

## claims 清单

- **P1**(Eq.7 直接推论):$\pi_\theta = \pi_{\mathrm{ref}}$ 时 $z=0$,损失恰为 $\log 2$(一切 DPO 实现的第一道冒烟测试)。
- **P2**(σ 单调性 + 成对差分):损失对 $z$ 严格单调递减;交换 $y_w/y_l$ 使 $z$ 反号,且恒等式 $L(y_l,y_w) - L(y_w,y_l) = z$ 成立(精确)。
- **P3**(Eq.5 往返):给定 $r$ 与 $\pi_{\mathrm{ref}}$,由 Eq.(4) 构造 $\pi^*$,再按 Eq.(5) 反演回奖励,应精确还原 $r$(配分函数项 $\beta\log Z$ 不能丢)。
- **P4**(Eq.4 归一化):$\sum_y \pi^*(y|x) = 1$(按构造的 $Z$ 归一)。
- **P5**:解析梯度 $\partial L/\partial(\log p)$ 结构:$\mp\beta\,\sigma(-z)$ 按四路分配,与有限差分一致。
- **不可测**(进 GAP_LIST):偏好数据上的期望、与 RLHF 基线的对比、$\beta$ 敏感性 —— 需要真实数据与训练。
