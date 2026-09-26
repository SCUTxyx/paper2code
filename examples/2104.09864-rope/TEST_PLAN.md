# 可测性规划:RoPE

| claim | 内容 | 测试类型 | 落点 | 备注 |
|---|---|---|---|---|
| R1 | 旋转正交:范数保持、$R^\top R=I$ | 性质-不变量 | test_properties.py::test_rotation_preserves_norm | 1e-12 |
| R2 | 相对位置不变性(核心 claim) | 性质-不变量 | test_properties.py::test_relative_position_invariance | 多组 (m,n,c),1e-12 |
| R3 | 复数形式 = 矩阵形式 | 对拍 | test_crosscheck.py | 双表述实现 |
| R4 | 复合律 $R^m R^n = R^{m+n}$ | 性质 | test_properties.py::test_composition | 1e-12 |
| R5 | 梯度 $R^{m\top}w$ | 梯度检查 | test_gradients.py | 含 score 对 q 的梯度 |
| — | d=2 手例:$f([1,0],3) = [\cos 3, \sin 3]$;$\langle f(q,3),f(k,1)\rangle = \sin 2$ | 锚点 | test_anchor.py | 标准三角常数,推导在注释 |
| §3.4.3 | 内积上界随相对距离衰减 | **无法自动判定** | → GAP_LIST #1 | 渐近上界,依赖向量结构 |
