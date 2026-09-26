# 公式—代码对照表:RoPE

| 论文位置 | 公式 | 实现位置 |
|---|---|---|
| Eq.(15) | $\theta_i = 10000^{-2(i-1)/d}$ | impl/core_eq.py:L14(`theta_seq`) |
| Eq.(13) | $f(q,m) = R^m_\Theta q$ | impl/core_eq.py:L17(`rope_rotate`) |
| Eq.(14) 块内 | $\begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}$ | impl/core_eq.py:L25-26 |
| 打分定义 | $\langle f(q,m), f(k,n)\rangle$ | impl/core_eq.py:L32 |
| Eq.(14) 显式矩阵 | 块对角 $R^m_\Theta$ | impl/core_pseudo.py:L12-24(`rotation_matrix`) |
| §3.2.1 复数形式 | $f(q,m) = q\,e^{\mathrm{i}m\theta}$ | impl/core_pseudo.py:L32-42 |
