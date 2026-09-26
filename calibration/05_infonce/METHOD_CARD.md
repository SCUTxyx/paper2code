# 方法卡片:InfoNCE / CLIP 对比损失

- 论文:① *Representation Learning with Contrastive Predictive Coding*(Oord et al., arXiv:1807.03748)§2.2 InfoNCE;② *Learning Transferable Visual Models From Natural Language Supervision*(Radford et al., arXiv:2103.00020)§2.3 Eq.(1)(2)。
- 范围声明:只覆盖批内对比损失本身;编码器、检索评测、batch 内假阳性处理不覆盖。

## 问题形式化

一个 batch 内 $N$ 对(图像 $I_i$, 文本 $T_i$)样本,把配对样本拉近、非配对推远。

## 符号表

| 符号 | 含义 | 出处 |
|---|---|---|
| $I_i, T_i$ | 第 $i$ 对图像 / 文本嵌入 | CLIP §2.3 |
| $\tau$ | 温度(学得或固定) | CLIP Eq.(1) |
| $\mathrm{sim}$ | 余弦相似度(嵌入先 L2 归一化) | CLIP §2.3 |
| $A$ | 注意力式相似度矩阵 $S/\tau$ | InfoNCE §2.2 |

## 核心公式

- CLIP Eq.(1):$L_{\mathrm{i2t}} = -\frac{1}{N}\sum_i^N \log\frac{\exp(\mathrm{sim}(I_i,T_i)/\tau)}{\sum_j^N \exp(\mathrm{sim}(I_i,T_j)/\tau)}$;$L_{\mathrm{t2i}}$ 对称。
- CLIP Eq.(2):$L = \frac{1}{2}(L_{\mathrm{i2t}} + L_{\mathrm{t2i}})$(双向平均)。
- InfoNCE §2.2:每行是「1 正样本 + N−1 负样本」的 softmax 交叉熵。

## claims 清单

- **N1**(Eq.2 结构):双向平均对塔交换不变:$L(I,T) = L(T,I)$。
- **N2**(InfoNCE §2.2 极限行为):$\tau\to\infty$ 时损失 → $\log N$(softmax → 均匀);正确配对严格占优时 $\tau\to 0$ 损失 → 0。
- **N3**(CLIP §2.3):嵌入内部 L2 归一化 → 损失对输入尺度不变。
- **N4**(批结构):对 batch 做联合置换(同置换 $I$ 与 $T$ 的行)损失不变。
- **N5**:解析梯度(含归一化的投影项)与有限差分一致。
