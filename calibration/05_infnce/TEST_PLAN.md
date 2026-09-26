# 可测性规划:InfoNCE / CLIP

| claim | 内容 | 测试类型 | 落点 | 备注 |
|---|---|---|---|---|
| N1 | L(I,T) = L(T,I) | 性质-不变量 | test_properties.py::test_tower_swap_symmetry | 精度 1e-12 |
| N2 | τ→∞ → log N;正确配对 τ→0 → 0 | 性质-极限 | test_properties.py::test_temperature_limits | |
| N3 | 输入尺度不变 | 性质-不变量 | test_properties.py::test_scale_invariance | |
| N4 | 批内联合置换不变 | 性质-不变量 | test_properties.py::test_batch_permutation_invariance | |
| N5 | 解析梯度(含归一化投影) | 梯度检查 | test_gradients.py | 对 I 与 T 两路 |
| — | 2×2 旋转构造手算值 | 锚点 | test_anchor.py | 推导在注释,常数落字面量;N=1 退化 L=0 |
| — | 向量化版 vs 逐对循环版 | 对拍 | test_crosscheck.py | |
