# Formula-to-code map: InfoNCE / CLIP

| Paper location | Formula | Implementation |
|---|---|---|
| CLIP §2.3 | embedding L2 normalization | impl/core_eq.py:L11-12 (`_normalize`) |
| CLIP Eq.(1) | logits $\mathrm{sim}/\tau$ | impl/core_eq.py:L39 |
| CLIP Eq.(1) | $L_{\mathrm{i2t}}$ and the symmetric $L_{\mathrm{t2i}}$ | impl/core_eq.py:L40-41 (CE in `_log_softmax_ce` L15-20) |
| CLIP Eq.(2) | $L = \frac{1}{2}(L_{\mathrm{i2t}}+L_{\mathrm{t2i}})$ | impl/core_eq.py:L42 |
| N5 analytic gradient | $\partial L/\partial S$ | impl/core_eq.py:L49 |
| N5 analytic gradient | $\partial L/\partial \hat I,\ \partial L/\partial \hat T$ (pre-projection) | impl/core_eq.py:L50-51 |
| N5 analytic gradient | normalization projection chain | impl/core_eq.py:L53-56 |
| InfoNCE §2.2 independent form | per-pair softmax (no matrix) | impl/core_pseudo.py:L12-29 (`clip_loss_loop`, per-pair softmax L22-24) |
