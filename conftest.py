"""仓库级 pytest 配置:把 scripts/ 加入 import 路径,使所有 repro 与考卷
都能 `from gradcheck import ...` 复用同一个有限差分工具。"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
