"""Repository-level pytest configuration: puts scripts/ on the import path so that
every repro and exam can reuse the shared finite-difference checker via
`from gradcheck import ...`."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
