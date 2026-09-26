"""Meta-test: every EQ_MAP.md line reference must point at a real, non-blank line.

EQ_MAP is verification-ladder level L4 (human review): a drifted line number
silently kills its value. This test parses all EQ_MAPs and re-checks every
`impl/core_*.py:L<n>` (and ranges, and bare `L<n>` tokens following a file
mention in the same row) against the actual files.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILE_RE = re.compile(r"impl/core_\w+\.py")
TOKEN_RE = re.compile(r"L(\d+)(?:-(\d+))?")


def _iter_map_files():
    yield from sorted((REPO / "calibration").glob("*/EQ_MAP.md"))
    yield from sorted((REPO / "examples").glob("*/EQ_MAP.md"))


def test_every_eq_map_reference_is_valid():
    checked = 0
    for eqmap in _iter_map_files():
        folder = eqmap.parent
        for line in eqmap.read_text().splitlines():
            current_file = None
            # walk the row left-to-right; a bare L-token belongs to the most
            # recent file mention in the same row
            for m in re.finditer(r"impl/core_\w+\.py|L\d+(?:-\d+)?", line):
                token = m.group(0)
                if FILE_RE.fullmatch(token):
                    current_file = folder / token
                    continue
                if current_file is None:
                    continue  # e.g. "Algorithm 1 L7-8" before any file mention
                for part in token[1:].split("-"):
                    n = int(part)
                    assert n >= 1, f"{eqmap}: bad line number {token}"
                    src = current_file.read_text().splitlines()
                    assert n <= len(src), (
                        f"{eqmap}: {current_file.name}:L{n} out of range "
                        f"(file has {len(src)} lines)")
                    assert src[n - 1].strip(), (
                        f"{eqmap}: {current_file.name}:L{n} points at a blank line "
                        f"— the map has drifted from the code")
                    checked += 1
    # guard against the checker silently matching nothing after a refactor
    assert checked >= 60, f"only {checked} references checked — did the EQ_MAP format change?"


def test_every_repro_folder_has_an_eq_map():
    for d in sorted((REPO / "examples").iterdir()) + sorted((REPO / "calibration").iterdir()):
        if d.is_dir():
            assert (d / "EQ_MAP.md").is_file(), f"{d.name} missing EQ_MAP.md"
