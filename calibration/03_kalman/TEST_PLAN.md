# 可测性规划:Kalman Filter

| claim | 内容 | 测试类型 | 落点 | 备注 |
|---|---|---|---|---|
| K1 | 递推后验均值 = 批量最小二乘(扩散先验) | 锚点(权威真值) | test_anchor.py::test_static_diffuse_prior_equals_lstsq | `np.linalg.lstsq` 是独立真值,非实现复述 |
| K2 | 后验均值 = GLS 闭式(信息先验) | 锚点 | test_anchor.py::test_informative_prior_gls_closed_form | |
| K3 | trace(P_t) 单调不增 | 性质-单调 | test_properties.py::test_covariance_shrinks | 静态模型 |
| K4 | 对观测的雅可比 = Λ_T⁻¹HᵀR⁻¹ | 梯度检查 | test_gradients.py | 信息形式解析 vs 中心差分 |
| K5 | NIS 期望 = d_y | 性质-统计(解析容差) | test_properties.py::test_nis_consistency | 蒙特卡洛,容差按 Var(NIS)=2d_y 导出 |
| — | 标准递推 vs 信息形式 | 对拍 | test_crosscheck.py | 含动态模型 F≠I、Q≠0 情形 |
