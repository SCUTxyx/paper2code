# 可测性规划:Adam

| claim | 内容 | 测试类型 | 落点 | 备注 |
|---|---|---|---|---|
| A1 | 首步步长 ≈ α 且与梯度尺度无关 | 性质-退化 | test_properties.py::test_first_step_size_independent_of_scale | |
| A2 | β→0 退化为符号型更新 α·g/(|g|+ε) | 性质-退化 | test_properties.py::test_beta_to_zero_degenerates | 对抗「ε 加错位置」类自洽误读,需与手写参照比 |
| A3 | 递推 = 历史加权和的闭式展开 | 对拍 + 梯度 | test_crosscheck.py / test_gradients.py | 双实现:递推(core_eq)vs 展开和(core_pseudo) |
| A4 | 常梯度下 m̂_t=g、v̂_t=g²(精确) | 性质-极限 | test_properties.py::test_constant_gradient_exact | |
| — | 前几步更新值闭式手算 | 锚点 | test_anchor.py | 推导写在注释,常数落字面量 |
| — | ∂m̂_t/∂g_k、∂v̂_t/∂g_k 解析系数 | 梯度检查 | test_gradients.py | 系数 (1−β)β^{t−k}/(1−β^t) 手写推导 |
