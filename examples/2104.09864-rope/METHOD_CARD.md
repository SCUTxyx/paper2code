# Method card: RoPE rotary position embedding (Su et al. 2021)

- Paper: *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864
- Scope statement: only the §3.2 rotary position embedding math piece (the formulas and
  their two equivalent formulations); the overall RoFormer architecture, the asymptotic
  decay upper bound (§3.4.3), and all experiments are not covered.

## Problem formalization

Inject **absolute position $m$** into query/key while making the inner product depend only
on the **relative position $m-n$**:
$$\langle f(q,m), f(k,n)\rangle = g(q,k,m-n)$$

## Symbol table

| Symbol | Meaning | Source |
|---|---|---|
| $q, k \in \mathbb{R}^d$ | attention query / key vectors | §3.2 |
| $m, n$ | absolute token positions | §3.2 |
| $\theta_i$ | rotation frequency of the $i$-th 2-dim block | Eq.(15) |
| $R^m_\Theta$ | $d\times d$ rotation matrix (block diagonal) | Eq.(14) |

## Core formulas (§3.2.2)

- Eq.(13):$f(q, m) = R^m_\Theta\, q$
- Eq.(14):$R^m_\Theta = \mathrm{diag}$ of blocks $\begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}$
- Eq.(15):$\theta_i = 10000^{-2(i-1)/d},\ i = 1,\dots,d/2$
- §3.2.1 complex formulation ($d=2$):$f(q, m) = q\, e^{\mathrm{i}m\theta}$ (q a complex number)

## Algorithm box (executable transcription of §3.2.2)

1. Precompute $\theta_1..\theta_{d/2}$ per Eq.(15);
2. For position $m$ and vector $x$, build $R^m_\Theta$ per Eq.(14) (or rotate block-wise);
3. Output $f(x, m) = R^m_\Theta x$.

## Claims list

- **R1** (Eq.14 rotation structure):$R^m_\Theta$ is orthogonal — $\|f(x,m)\| = \|x\|$ and
  $R^{m\top}R^m = I$ (exact).
- **R2** (core claim, §3.2.2):relative-position invariance
  $\langle f(q,m), f(k,n)\rangle = \langle f(q,m+c), f(k,n+c)\rangle$ for any $c$ (exact, 1e-12).
- **R3** (§3.2.1 ↔ §3.2.2):the complex and matrix formulations agree for the same
  $(x, m)$ (exact).
- **R4** (rotation group structure):$R^m R^n = R^{m+n}$ — rotating by $m$ then $n$ equals
  rotating by $m+n$ (exact).
- **R5**:gradient $\partial\, w^\top f(q,m)/\partial q = R^{m\top} w$ (the rotation's
  transpose is the inverse rotation).
- **Not testable** (→ GAP_LIST):the §3.4.3 decay of the inner-product upper bound with
  relative distance is an asymptotic statement depending on vector structure; it cannot
  become a deterministic test.
