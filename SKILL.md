---
name: paper2code
description: 复现论文核心方法并验证。输入 arXiv 链接或论文文件,产出 numpy 最小实现 + 全绿的梯度/性质测试 + 方法卡片 + 公式对照表 + 诚实的复现差距清单。触发词:复现这篇论文、把论文变成代码、实现论文方法、验证论文公式、reproduce this paper、paper to code、论文复现、给我这篇论文方法的最小实现。
---

# paper2code:把「复现一篇论文的核心方法」变成有验证的标准动作

**原则:验证数学,不跑训练。** 输入一篇论文,输出核心模块的 numpy 最小实现
+ 全绿测试 + 方法卡片 + 公式对照表 + 差距清单。全程自动、零重依赖(torch/jax/tf 一律禁止)。

**何时不用本 skill**:用户要的是「跑通官方仓库」「复现论文表格里的 SOTA 数字」
「端到端训练」——这些超出范围(缺数据/GPU/训练时长),如实说明后可只做其中可测的数学件。

## 产出契约(每次运行固定,不得增删改名)

在仓库工作目录创建 `repros/<arxiv-id>-<short-name>/`,七个文件:

```
METHOD_CARD.md        # 方法卡片:问题形式化、符号表、核心公式(带原文引用)、算法框、claims 清单
TEST_PLAN.md          # 每条 claim → 测试类型的映射(可审计)
impl/core_eq.py       # 按论文公式逐条实现的版本
impl/core_pseudo.py   # 按伪代码/等价表述独立实现的版本(可得时)
tests/test_gradients.py / test_properties.py / test_anchor.py / test_crosscheck.py
REPORT.md             # 验证报告:结果矩阵、发现(含论文疑点)、已知局限
GAP_LIST.md           # 复现差距清单:每项写明缺什么(数据/GPU/训练时长)
EQ_MAP.md             # 公式—代码对照表:公式号 → 文件:行
```

骨架模板见 `references/templates.md`,验证阶梯与容差细则见 `references/verification.md`。

## 六阶段流水线

### S0 获取与解析
1. arXiv 论文优先取 ar5iv HTML(`https://ar5iv.labs.arxiv.org/html/<id>`),整页读入;
   HTML 拿不到再读 PDF(分页,每次 ≤ 20 页)。
2. 定位核心贡献章节(通常是 §3 method 及其依赖的附录)。**精读只限这一节 + 相关附录**,
   其余部分只浏览标题——这是 token 成本的主要控制点(全文浏览限 2 次)。
3. 系统型/数据集型论文在此做适用性预检(见下)。

### S1 方法抽取 → METHOD_CARD.md
- 符号表:每个符号的含义 + 出处公式号;
- 公式清单:逐条 LaTeX 原文引用(编号用论文编号);
- **claims 列表**:论文声称的每一条性质——退化关系、不变量、极限行为、方向性/单调性声明;
- 算法框原文(逐行)。
- 「论文声称 X」必须标注出处(§/Eq.)。claims 找不全,S2 就会漏测。

### S2 可测性规划 → TEST_PLAN.md
把每条 claim 映射到测试类型(一张表):
- 解析可验 → 梯度检查 / 双实现对拍;
- 声明退化 → 退化测试(参数极端值还原为已知方法);
- 声明不变量 → 不变量测试(归一化、置换、平移);
- 论文有 toy 数字/闭式小例 → 锚点测试;
- 判不了 → 如实进 GAP_LIST,写明缺什么。
**适用性预检**:系统型论文只测其中可测的数学件(调度公式、通信量模型等),
数据集型论文只测统计口径一致性,其余全部进 GAP_LIST——诚实边界是本 skill 的核心价值。

### S3 最小实现
硬约束(违反任何一条即返工):
1. **numpy-only**:禁止 import torch/jax/tf/sklearn;
2. 合成数据,显式播种 `np.random.default_rng(seed)`;
3. **先引用公式再实现**:函数 docstring 首行引用公式号,行内注释标公式号(保证 EQ_MAP 可生成);
4. float64;
5. 能拿到两个独立表述(公式 vs 伪代码/等价推导)就做双实现 `core_eq.py` + `core_pseudo.py`,
   **两者不得互相 import**,计算路径必须不同(供互对拍,防线见 verification.md §3)。

### S4 验证执行
- 四级阶梯(细则见 verification.md §1):L1 参考对拍 → L2 性质测试 → L3 论文锚点 → L4 EQ_MAP。
- 四类测试文件全部要有;梯度检查一律复用 `scripts/gradcheck.py`(float64 中心差分,
  步长 cbrt(eps)·max(1,|x|),rtol<1e-6);统计型容差必须按 5σ 解析导出,禁止拍脑袋。
- 运行:`python -m pytest repros/<dir>/tests -q`。**测试不过不许写「验证通过」;
  fail 的测试如实进 REPORT——失败本身是发现**(可能是实现错、论文笔误、或声明过强)。

### S5 报告生成
- REPORT.md:结果矩阵(每个文件的用例数/通过数/最大误差)+ 发现(含论文疑点)+ 已知局限;
- GAP_LIST.md:每项写明缺什么(数据/GPU/训练时长/信息不足)与影响;
- EQ_MAP.md:公式号 → 文件:行(写完代码后用 grep 取行号,不得手估);
- 最后把整个 repro 目录给用户看一遍,提示 EQ_MAP 供人工复核——**最终裁决权在人**。

## 复用与校准

- 通用工具:`scripts/gradcheck.py`(所有 repro 复用);
- 考卷:`calibration/` 下 5 篇「正确答案已知」的论文(Adam/Attention/Kalman/DDPM/InfoNCE),
  同时是产物模板的活样例;
- **每次修改本 SKILL.md 或 scripts/ 后,必须跑 `bash scripts/run_calibration.sh`,
  5/5 全绿才算有效变更**(结果自动追加到 CALIBRATION_LOG.md)。

## 诚实条款

1. 产物结构固定——固定性就是可靠性;
2. 判不了的 claim 进 GAP_LIST,不许沉默省略;
3. 论文本身的笔误/前后不一致记入 REPORT「发现」——这是本 skill 区别于「实现机器」的独特产出;
4. 所有统计容差可追溯到解析推导;
5. 报告是验证报告,不是正确性保证书。
