# 公式—代码对照表:Attention

| 论文位置 | 公式 | 实现位置 |
|---|---|---|
| Eq.(1) | $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ | impl/core_eq.py:L28(打分)、L31(softmax)、L32(加权) |
| §3.2.3 掩码 | 未来位置 logit = $-\infty$ | impl/core_eq.py:L29-30(`causal_mask` L18-19) |
| softmax 反传(T5) | $A\odot(\partial A - \mathrm{rowsum}(\cdot))$ | impl/core_eq.py:L45-46 |
| Eq.(1) 独立表述 | 逐查询、允许集合内显式归一化 | impl/core_pseudo.py:L21-31 |
