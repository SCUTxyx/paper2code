# Formula-to-code map: Attention

| Paper location | Formula | Implementation |
|---|---|---|
| Eq.(1) | $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ | impl/core_eq.py:L28 (scores), L31 (softmax), L32 (weighted sum) |
| §3.2.3 mask | future-position logits $=-\infty$ | impl/core_eq.py:L29-30 (`causal_mask` at L17-19) |
| T5 backward | softmax backward $A\odot(\partial A - \mathrm{rowsum}(\cdot))$ | impl/core_eq.py:L38-44 (`attention_grads`, dS at L40) |
| Eq.(1) independent form | per-query loop, explicit within-allowed-set normalization | impl/core_pseudo.py:L13-29 (`attention_loop`, normalization at L27) |
