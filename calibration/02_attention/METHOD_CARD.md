# 方法卡片:Scaled Dot-Product Attention(Vaswani et al. 2017)

- 论文:*Attention Is All You Need*,arXiv:1706.03762(NeurIPS 2017)
- 范围声明:只覆盖 §3.2.1 的 scaled dot-product attention(含因果掩码);多头拼接、位置编码、整体 Transformer 不覆盖。

## 问题形式化

给定查询序列 $Q\in\mathbb{R}^{n\times d_k}$、键 $K\in\mathbb{R}^{n\times d_k}$、值 $V\in\mathbb{R}^{n\times d_v}$,输出与 $Q$ 等长的加权和序列。

## 符号表

| 符号 | 含义 | 出处 |
|---|---|---|
| $Q, K, V$ | 查询 / 键 / 值矩阵 | §3.2.1 |
| $d_k$ | 键维度(缩放因子 $\sqrt{d_k}$ 的来源) | §3.2.1 |
| $A$ | 注意力权重 $\mathrm{softmax}(QK^\top/\sqrt{d_k})$ | Eq.(1) |
| $M$ | 因果掩码(位置 $j>i$ 禁止注意) | §3.2.3 decoder 自回归需要 |

## 核心公式

- Eq.(1):$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(QK^\top/\sqrt{d_k})\,V$
- 因果掩码变体:softmax 前把 $j>i$ 的 logit 置 $-\infty$(§3.2.3「ensures that predictions for position $i$ can depend only on the known outputs at positions less than $i$」)

## claims 清单

- **T1**(Eq.1 softmax 定义):注意力权重每行和为 1,每行是合法凸组合。
- **T2**(§3.2.3 掩码声明):因果掩码下位置 $i$ 的输出与位置 $j>i$ 的值无关 —— 权重 $A_{ij}=0$(精确),且 $\partial\,\mathrm{out}_i/\partial V_j = 0$ 对所有 $j>i$。注意全损失对 $V_j$ 的梯度不为零($\partial L/\partial V_j = \sum_{i\ge j}A_{ij}W_i$),为零的是「单输出行对 $V_j$」的依赖。
- **T3**(§3.2.1 缩放动机):对 $q,k\sim\mathcal N(0,I)$,点积方差 $\approx d_k$;$1/\sqrt{d_k}$ 缩放后方差 $\approx 1$,避免 softmax 进入饱和区。
- **T4**(Eq.1 结构):无掩码时注意力对查询是逐行映射 —— 置换查询行,输出行做同样置换(置换等变)。
- **T5**:梯度可解析给出:$\partial L/\partial A = W V^\top$、softmax 反传 $A\odot(\partial L/\partial A - \mathrm{rowsum}(A\odot\partial L/\partial A))$、$\partial L/\partial V = A^\top W$。
