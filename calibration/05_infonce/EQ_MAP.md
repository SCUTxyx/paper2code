# Formula-to-code map: InfoNCE / CLIP

| Paper location | Formula | Implementation |
|---|---|---|
| CLIP §2.3 | embedding L2 normalization | impl/core_eq.py:L11-12 (`_normalize`) |
| CLIP Eq.(1) | logits $\mathrm{sim}/\tau$ | impl/core_eq.py:L38 |
| CLIP Eq.(1) | $L_{\mathrm{i2t}}$ and the symmetric $L_{\mathrm{t2i}}$ | impl/core_eq.py:L39-40 (CE in `_log_softmax_ce` L15-19) |
| CLIP Eq.(2) | $L = \frac{1}{2}(L_{\mathrm{i2t}}+L_{\mathrm{t2i}})$ | impl/core_eq.py:L41 |
| N5 analytic gradient | $\partial L/\partial S$ | impl/core_eq.py:L48 |
| N5 analytic gradient | $\partial L/\partial \hat I,\ \partial L/\partial \hat T$ (pre-projection) | impl/core_eq.py:L49-50 |
| N5 analytic gradient | normalization projection chain | impl/core_eq.py:L52-55 |
| InfoNCE §2.2 independent form | per-pair softmax (no matrix) | impl/core_pseudo.py:L12-29 (`clip_loss_loop`, per-pair softmax L22-24) |
