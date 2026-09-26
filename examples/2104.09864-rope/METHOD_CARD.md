# 方法卡片:RoPE 旋转位置编码(Su et al. 2021)

- 论文:*RoFormer: Enhanced Transformer with Rotary Position Embedding*,arXiv:2104.09864
- 范围声明:只覆盖 §3.2 的旋转位置编码(公式与两种等价表述);RoFormer 整体架构、衰减上界的渐近分析(§3.4.3)、实验不覆盖。

## 问题形式化

给 query/key 注入**绝对位置 $m$**,同时使内积只依赖**相对位置 $m-n$**:
$$\langle f(q,m), f(k,n)\rangle = g(q,k,m-n)$$

## 符号表

| 符号 | 含义 | 出处 |
|---|---|---|
| $q, k \in \mathbb{R}^d$ | 注意力的 query / key 向量 | §3.2 |
| $m, n$ | token 的绝对位置 | §3.2 |
| $\theta_i$ | 第 $i$ 个二维块的旋转角频率 | Eq.(15) |
| $R^m_\Theta$ | $d\times d$ 旋转矩阵(块对角) | Eq.(14) |

## 核心公式(§3.2.2)

- Eq.(13):$f(q, m) = R^m_\Theta\, q$
- Eq.(14):$R^m_\Theta = \mathrm{diag}$ 块,第 $i$ 块为 $\begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}$
- Eq.(15):$\theta_i = 10000^{-2(i-1)/d},\ i = 1,\dots,d/2$
- §3.2.1 复数表述($d=2$):$f(q, m) = q\, e^{\mathrm{i}m\theta}$(q 为复数)

## 算法框(论文 §3.2.2 的可执行转写)

1. 按 Eq.(15) 预计算 $\theta_1..\theta_{d/2}$;
2. 对位置 $m$ 的向量 $x$,按 Eq.(14) 构造 $R^m_\Theta$(或按块直接旋转);
3. 输出 $f(x, m) = R^m_\Theta x$。

## claims 清单

- **R1**(Eq.14 旋转结构):$R^m_\Theta$ 正交 —— $\|f(x,m)\| = \|x\|$ 且 $R^{m\top}R^m = I$(精确)。
- **R2**(§3.2.2 核心声明):相对位置不变性 $\langle f(q,m), f(k,n)\rangle = \langle f(q,m+c), f(k,n+c)\rangle$ 对任意 $c$(精确,1e-12)。
- **R3**(§3.2.1 ↔ §3.2.2):复数表述与矩阵表述对同一 $(x, m)$ 给出相同结果(精确)。
- **R4**(旋转群结构):$R^m R^n = R^{m+n}$ —— 先转 $m$ 再转 $n$ = 直接转 $m+n$(精确)。
- **R5**:梯度解析:$\partial\, w^\top f(q,m)/\partial q = R^{m\top} w$(旋转矩阵的转置即逆旋转)。
- **不可测**(进 GAP_LIST):§3.4.3 的内积上界随相对距离**衰减**是渐近上界声明,依赖向量具体结构,无法转化为确定性测试。
