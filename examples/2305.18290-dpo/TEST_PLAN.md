# 可测性规划:DPO

| claim | 内容 | 测试类型 | 落点 | 备注 |
|---|---|---|---|---|
| P1 | π=π_ref 时 L = log 2(精确) | 性质-退化 | test_properties.py::test_reference_policy_gives_log2 | |
| P2 | 对 z 单调递减;交换恒等式 L(l,w)−L(w,l)=z | 性质-单调/不变量 | test_properties.py::test_monotone + test_swap_identity | 交换恒等式 1e-12 |
| P3 | Eq.(5) 奖励重参数化往返 | 对拍(闭式恒等) | test_crosscheck.py::test_reward_roundtrip | β·logZ 项的作用点 |
| P4 | π* 归一化 | 性质-不变量 | test_crosscheck.py::test_optimal_policy_normalized | |
| P5 | 解析梯度 | 梯度检查 | test_gradients.py | 4 路 log-prob 输入 |
| — | z=ln3 → L=ln(4/3);z=0 → log 2;z=−ln3 → ln 4 | 锚点 | test_anchor.py | 手算精确常数,推导在注释 |
| — | Eq.(7) 直写 vs Eq.(6) BT 路径 | 对拍 | test_crosscheck.py::test_eq7_vs_bradley_terry | 双表述实现 |
| 数据期望 | −E[·] 在偏好分布上 | **无法自动判定** | → GAP_LIST #1 | 需真实偏好数据 |
