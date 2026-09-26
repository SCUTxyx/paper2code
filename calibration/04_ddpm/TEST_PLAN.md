# 可测性规划:DDPM 前向过程

| claim | 内容 | 测试类型 | 落点 | 备注 |
|---|---|---|---|---|
| D1 | 闭式边缘 = 逐步加噪的分布(蒙卡) | 锚点(统计型) | test_anchor.py | 容差 5σ/√N 由 N 与 ᾱ_t 解析导出 |
| D2 | 同噪声下递推 = 未展开和式(精确) | 对拍 | test_crosscheck.py | 两条独立计算路径,容差 1e-10 |
| D3 | ᾱ 严格下降;ᾱ_T→0;x_T→N(0,I) | 性质-单调/极限 | test_properties.py | |
| D4 | 对 x₀/ε/ε_s 的梯度 | 梯度检查 | test_gradients.py | 中心差分 vs 解析系数 |
| — | 调度端点 β₁=1e-4、β_T=0.02 | 锚点 | test_properties.py | §4 原文数值 |
