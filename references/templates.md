# 产物模板(产出契约的七个文件)

每次运行的产物结构完全固定 —— 这份固定性就是 skill 可靠性的来源。
目录名:`repros/<arxiv-id>-<short-name>/`(示例:`2104.09864-rope/`)。

```
repros/<arxiv-id>-<short-name>/
├── METHOD_CARD.md        # ① 方法卡片
├── TEST_PLAN.md          # ② 可测性规划
├── impl/
│   ├── core_eq.py        # ③ 按论文公式逐条实现
│   └── core_pseudo.py    #    按伪代码/等价表述独立实现(可得时)
├── tests/
│   ├── test_gradients.py # ④ 四类测试(文件名固定)
│   ├── test_properties.py
│   ├── test_anchor.py
│   └── test_crosscheck.py
├── REPORT.md             # ⑤ 验证报告
├── GAP_LIST.md           # ⑥ 复现差距清单
└── EQ_MAP.md             # ⑦ 公式—代码对照表
```

## ① METHOD_CARD.md 骨架

```markdown
# 方法卡片:<论文名>
- 论文:<标题>, arXiv:<id>(<venue 年份>)
- 范围声明:本卡片只覆盖「核心方法节」;其余部分明确声明不覆盖。

## 问题形式化
<输入 / 输出 / 目标,两三句>

## 符号表
| 符号 | 含义 | 出处 |
|---|---|---|
| x_t | … | Eq.(n) / §m |

## 核心公式(逐条,LaTeX 原文引用)
- Eq.(n):$...$ —— <一句话解释>

## 算法框
<Algorithm n 逐行原文或忠实转写>

## claims 清单(论文声称的每一条性质)
- <编号> <声明内容>(出处 §m / Eq.n)
```

## ② TEST_PLAN.md 骨架

每条 claim 一行,映射到测试类型;判不了的写明去向。

```markdown
| claim | 内容 | 测试类型 | 落在哪个文件 | 备注 |
|---|---|---|---|---|
| C1 | … | 性质-不变量 | test_properties.py::test_x | … |
| C5 | … | 无法自动判定 | → GAP_LIST #2 | 需要真实数据 |
```

## ③④ 实现与测试规范

- `core_eq.py`:按论文公式逐条实现;每个函数 docstring 首行引用公式号;行内注释标公式号(保证 EQ_MAP 可生成)。
- `core_pseudo.py`:**独立表述**(伪代码直译 / 等价闭式 / 循环化),不得 import `core_eq`。
- numpy-only、float64;随机数 `np.random.default_rng(seed)`。
- 测试文件标准头(跨 repro 同名 impl 模块隔离):

```python
import sys
from pathlib import Path
IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]          # 同名 impl 模块跨 repro 冲突防护
import core_eq
```

- 梯度检查一律走 `from gradcheck import assert_grad_close`(scripts/ 已在仓库 conftest.py 加入路径)。

## ⑤ REPORT.md 骨架

```markdown
# 复现报告:<short-name>(arXiv:<id>)

## 结果矩阵
| 测试文件 | 用例数 | 通过 | 失败 | 最大相对误差/备注 |
|---|---|---|---|---|

## 发现(含论文疑点)
1. …

## 已知局限
- …
```

## ⑥ GAP_LIST.md 骨架

每项写明**缺什么**(数据 / GPU / 训练时长 / 信息不足),以及影响。

```markdown
| # | 差距 | 缺什么 | 影响 |
|---|---|---|---|
| 1 | … | 真实数据集 | 效果数字未验证,仅验证数学件 |
```

## ⑦ EQ_MAP.md 骨架

```markdown
| 论文公式 | 内容摘要 | 实现位置 |
|---|---|---|
| Eq.(7) | $...$ | impl/core_eq.py:L<n> |
| Algorithm 1 L3 | $m_t = ...$ | impl/core_eq.py:L<n> |
```

## 运行方式

- 单个 repro:`python -m pytest repros/<dir>/tests -q`
- 全量(考卷 + 示例):`bash scripts/run_all_tests.sh`
- skill 回归(改 SKILL.md 后必跑):`bash scripts/run_calibration.sh`
