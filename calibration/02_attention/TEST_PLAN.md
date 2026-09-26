# 可测性规划:Attention

| claim | 内容 | 测试类型 | 落点 | 备注 |
|---|---|---|---|---|
| T1 | softmax 行和 = 1 | 性质-不变量 | test_properties.py::test_rows_sum_to_one | 精度 1e-12 |
| T2 | 掩码未来位置权重 0;∂out_i/∂V_j = 0 (j>i) | 性质 + 梯度 | test_properties.py::test_causal_mask_zero + test_gradients.py::test_causal_row_grad_wrt_V | 全损失对 V_j 的梯度不为零,见 METHOD_CARD T2 注 |
| T3 | 点积方差 ≈ d_k,缩放后 ≈ 1 | 性质-统计(解析容差) | test_properties.py::test_scaling_variance | 播种蒙特卡洛,容差按 3σ 解析导出 |
| T4 | 查询置换等变 | 性质-不变量 | test_properties.py::test_query_permutation_equivariance | 仅无掩码情形 |
| T5 | 解析梯度 | 梯度检查 | test_gradients.py | Q/K/V 三路,含因果掩码 |
| — | d=1 两键手例(sigmoid 恒等式) | 锚点 | test_anchor.py | softmax{1,-1} = sigmoid(2) 精确常数 |
| — | 公式版 vs 逐查询循环版 | 对拍 | test_crosscheck.py | 两条独立计算路径 |
