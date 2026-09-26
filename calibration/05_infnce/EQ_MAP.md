# 公式—代码对照表:InfoNCE / CLIP

| 论文位置 | 公式 | 实现位置 |
|---|---|---|
| CLIP §2.3 | 嵌入 L2 归一化 | impl/core_eq.py:L9-10(`_normalize`) |
| CLIP Eq.(1) | logits $\mathrm{sim}/\tau$ | impl/core_eq.py:L37 |
| CLIP Eq.(1) | $L_{\mathrm{i2t}}$ 与对称的 $L_{\mathrm{t2i}}$ | impl/core_eq.py:L38-39(CE 实现 L13-18) |
| CLIP Eq.(2) | $L = \frac{1}{2}(L_{\mathrm{i2t}}+L_{\mathrm{t2i}})$ | impl/core_eq.py:L40 |
| N5 解析梯度 | $\partial L/\partial S$、归一化投影 | impl/core_eq.py:L47-57 |
| InfoNCE §2.2 独立表述 | 逐对 softmax(不构造矩阵) | impl/core_pseudo.py:L16-27 |
