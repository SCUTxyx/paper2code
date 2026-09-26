# paper2code

> 把「复现一篇论文的核心方法」变成一个有验证、有边界声明的标准动作。
> **原则:验证数学,不跑训练。**

输入一篇 arXiv 论文,输出:核心模块的 **numpy 最小实现** + **全绿的梯度/性质测试**
+ 方法卡片 + 公式—代码对照表 + **诚实的复现差距清单**。以 agent skill 形态交付,
全程自动,零重依赖。

## 为什么

| 现有方式 | 给什么 | 缺什么 |
|---|---|---|
| Papers With Code | 官方/社区实现的索引 | 不验证,非按需 |
| labml.ai annotated | 人工精选的精读实现 | 覆盖少,更新慢 |
| 直接问 LLM | 快 | 无验证,无边界声明 |

paper2code 的差异 = **可验证**(性质 + 梯度 + 对拍 + 锚点)+ **诚实边界**(差距清单)
+ **agent 原生**(SKILL.md,全自动)+ **零重依赖**(numpy-only)。

## 快速开始(≤ 5 分钟,零新依赖)

```bash
git clone https://github.com/SCUTxyx/paper2code.git
cd paper2code
pip install numpy pytest          # conda test 环境已具备则跳过
bash scripts/run_all_tests.sh     # 校准集 5/5 + 两篇示例 repro 全绿
```

安装为 agent skill:把 `SKILL.md` 放进你的 skill 目录(如 `~/.zcode/skills/paper2code/SKILL.md`),
然后对 agent 说「**复现这篇论文:<arXiv 链接>**」。

## 验证体系:四级阶梯

| 级别 | 手段 | 适用 |
|---|---|---|
| L1 对拍 | 与权威参考实现同输入比输出(rtol=1e-5) | 有参考时最强 |
| L2 性质测试 | 退化 / 不变量 / 梯度检查(rtol<1e-6)/ 极限与单调性 | 无参考时的主力 |
| L3 论文锚点 | 复算论文自带的 toy 数字/闭式小例 | 唯一能直接对拍作者的机会 |
| L4 公式对照表 | 公式号 → 文件:行 | 不自动验证,把人审成本降到几分钟 |

残余风险与防线:**L2 抓不住「错得自洽」的误读**(如 ε 加错位置、性质照样全绿)。
防线 = 双表述独立实现互对拍 + EQ_MAP 人工复核;**最终裁决权始终留给人**。

## 校准集:skill 自己的回归测试(5/5 全绿)

5 篇「正确答案已知」的论文当考卷,每篇完整跑一遍,真值全部 numpy 可得:

| 考卷 | 校准点 | 测试数 |
|---|---|---|
| Adam | 前几步更新闭式手算;偏差修正首步 ≈ lr;β→0 退化 | 10 ✅ |
| Attention | softmax 行和;因果掩码未来位置零依赖;1/√d 缩放 | 10 ✅ |
| Kalman Filter | 静态线性高斯下递推后验 = 批量最小二乘(`lstsq`) | 7 ✅ |
| DDPM | q(x_t\|x₀) 闭式边缘 vs 逐步加噪蒙特卡洛 | 10 ✅ |
| InfoNCE/CLIP | 双向对称;温度极限退化;置换/尺度不变 | 10 ✅ |

每次修改 SKILL.md 后重跑 `bash scripts/run_calibration.sh`,结果追加到
[CALIBRATION_LOG.md](CALIBRATION_LOG.md)。构建考卷过程中的两个数值/口径发现
(Kalman 扩散先验的灾难性消去、Attention 因果性质的常见错误表述)记录在
[calibration/README.md](calibration/README.md)——它们本身就是本方法有效性的证据。

## 示例产物(真实论文试跑)

每份 repro 固定七件套:`METHOD_CARD / TEST_PLAN / impl(双实现) / tests(四类) / REPORT / GAP_LIST / EQ_MAP`。

- **[RoPE 旋转位置编码](examples/2104.09864-rope/)**(arXiv:2104.09864)— 13 测试全绿;
  发现 1:θ 的 1-based/0-based 索引约定错位是真实实现的首要风险点,性质测试抓不住,只有 EQ_MAP 能兜住;
  差距 1:§3.4.3 衰减上界是渐近声明,无法转化为确定性测试 → 如实进 GAP_LIST。
- **[DPO 直接偏好优化](examples/2305.18290-dpo/)**(arXiv:2305.18290)— 11 测试全绿;
  发现 1:必须用 log-prob 差而非概率比(数值稳定性);发现 3:Eq.(5) 的 β·logZ 项在成对差分中消去、在奖励反演中不可丢(构造反例验证)。

REPORT 节选(DDPM 校准考卷的锚点测试,容差按 5σ/√N 解析导出):

```
test_closed_form_vs_iterative_monte_carlo  闭式采样与逐步加噪在 t=400 处
                                           均值/方差均落在解析容差内        PASS
test_iterative_equals_unrolled             同噪声序列下递推 = 未展开和式     PASS
                                           (精确恒等,atol=1e-10)
```

## 成本

| 维度 | 单次复现 | 说明 |
|---|---|---|
| token | 50–150K(有上界) | 精读限定方法节,全文浏览限次 |
| 本地 CPU | < 10 秒 | 小矩阵合成数据测试 |
| 存储 | 30–100 KB / 篇 | 无数据集、无权重、无缓存 |
| 依赖 | 零新增 | numpy / pytest |

## 仓库结构

```
SKILL.md              # skill 本体(六阶段流水线 + 硬约束)
references/           # 验证阶梯细则、数值规范、产物模板
scripts/              # gradcheck.py(通用梯度检查)、一键回归脚本
calibration/          # 5 篇考卷(同时是产物模板)
examples/             # 真实论文试跑产物(RoPE、DPO)
repros/               # 用户本地运行区(.gitignore)
CALIBRATION_LOG.md    # 校准回归记录
PLAN.md               # 项目规划(九节:概述/契约/方法论/验证/里程碑/仓库/成本/风险/标准)
```

## 条款

- 仓库只放代码与报告,不放论文 PDF(arXiv 链接引用);MIT 许可。
- 欢迎按 `references/templates.md` 的模板贡献新 repro:一个 PR = 一篇论文的
  七件套 + 全绿测试;论文疑点与失败案例同样欢迎——它们是本仓库最有价值的部分。

详细方法论见 [PLAN.md](PLAN.md) 与 [SKILL.md](SKILL.md)。
