# 校准集:5 篇「正确答案已知」的考卷

这是 skill 自身的回归测试(PLAN §4.3)。每次修改 SKILL.md 或共享脚本后,
必须重跑 `bash scripts/run_calibration.sh`,结果追加到根目录 `CALIBRATION_LOG.md`。
**通过标准:5/5 全绿。** 单次执行秒级。

| 考卷 | 论文 | 校准点 | 真值来源(全部 numpy 可得) |
|---|---|---|---|
| 01_adam | Adam(Kingma & Ba 2015) | 前几步更新闭式手算;偏差修正首步步长 ≈ lr;β→0 退化 | 手算参照值 + 双实现 |
| 02_attention | Attention(Vaswani 2017) | softmax 行和;因果掩码未来位置零依赖;1/√d 缩放 | 性质 + 双实现 + sigmoid 恒等式锚点 |
| 03_kalman | Kalman Filter(1960) | 静态线性高斯下递推后验 = 批量最小二乘 | `np.linalg.lstsq`(零额外依赖) |
| 04_ddpm | DDPM(Ho et al. 2020) | q(x_t\|x₀) 闭式边缘 vs 逐步加噪蒙卡 | 闭式 vs 蒙特卡洛 |
| 05_infnce | InfoNCE / CLIP | 双向对称性;温度极限退化;置换/尺度不变 | 性质 + 退化 + 手算锚点 |

## 每份考卷的组成

与产出契约同构(固定产物结构),因此考卷同时是产物模板:
`METHOD_CARD.md`、`TEST_PLAN.md`、`impl/core_eq.py` + `impl/core_pseudo.py`
(双表述实现)、`tests/` 四类测试、`EQ_MAP.md`。

## 记录在案的发现(来自构建考卷本身)

- **Kalman(03)**:教科书协方差更新 $(I-KH)P^-$ 在扩散先验 $P_0=10^{10}$ 时
  因灾难性消去产生负定协方差,后验均值偏离批量最小二乘达 1e-1;换成代数恒等的
  Joseph 形式(1968)后,$P_0=10^8$ 时误差 2.4e-8。这正是「性质测试全绿也可能
  数值失效」的活例子 —— 已写进 `03_kalman/tests/test_anchor.py` 的文档字符串。
- **Attention(02)**:「全损失对 $V_j$ 的梯度为零」是**错误**的性质表述
  (后面的 query 会看 $V_j$);正确的因果性质是逐行的
  $\partial\,\mathrm{out}_i/\partial V_j = 0\ (j>i)$。性质表述本身需要被验证,
  这就是 TEST_PLAN 要逐条写清测试口径的原因。

## 数值规范

梯度检查用 `scripts/gradcheck.py`(float64 中心差分,步长 cbrt(eps)·max(1,|x|),
rtol < 1e-6);值对拍 rtol = 1e-5;精确关系 1e-12;统计型容差一律按 5σ 解析导出。
