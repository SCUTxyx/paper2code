"""Meta-test: each example REPORT's claimed case counts must equal what pytest
actually collects in that folder.

Keeps the published results matrices honest: if a test is added or removed
without updating the REPORT, or a REPORT inflates its numbers, this fails.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _collected_count(tests_dir: Path) -> int:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests_dir), "--collect-only", "-q"],
        capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stdout + out.stderr
    m = re.search(r"(\d+) tests collected", out.stdout)
    assert m, f"cannot parse collection output for {tests_dir}"
    return int(m.group(1))


def test_report_case_counts_match_reality():
    for report in sorted((REPO / "examples").glob("*/REPORT.md")):
        folder = report.parent
        claimed = 0
        for line in report.read_text().splitlines():
            if line.startswith("| test_"):
                cells = [c.strip() for c in line.split("|")]
                claimed += int(cells[2])  # "| test_x.py | cases | pass | fail |"
        actual = _collected_count(folder / "tests")
        assert claimed == actual, (
            f"{folder.name}: REPORT claims {claimed} cases but pytest collects {actual} — "
            f"update the results matrix")
        # every claimed row must be all-green (pass == cases)
        for line in report.read_text().splitlines():
            if line.startswith("| test_"):
                cells = [c.strip() for c in line.split("|")]
                assert cells[3] == cells[2], (
                    f"{folder.name}: REPORT row {cells[1]} claims failures "
                    f"({cells[3]} pass / {cells[2]} cases) — REPORT must reflect reality")


def test_calibration_log_latest_entry_is_all_green():
    log = (REPO / "CALIBRATION_LOG.md").read_text()
    blocks = log.split("## ")[1:]
    assert blocks, "CALIBRATION_LOG.md has no entries"
    latest = blocks[-1]
    assert "all green" in latest.splitlines()[0], (
        "latest calibration entry is not all green — rerun scripts/run_calibration.sh")
