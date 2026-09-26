"""Meta-test: the artifact contract is enforced structurally.

Every repro under examples/ carries the full seven-artifact contract (plus the
two report files); every calibration exam carries the nine files of the contract
minus REPORT/GAP_LIST (run artifacts live in CALIBRATION_LOG.md instead).
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CONTRACT = [
    "METHOD_CARD.md",
    "TEST_PLAN.md",
    "EQ_MAP.md",
    "impl/core_eq.py",
    "impl/core_pseudo.py",
    "tests/test_gradients.py",
    "tests/test_properties.py",
    "tests/test_anchor.py",
    "tests/test_crosscheck.py",
]


def test_examples_carry_the_full_contract():
    folders = [d for d in (REPO / "examples").iterdir() if d.is_dir()]
    assert len(folders) >= 5, "expected at least five example repros"
    for d in folders:
        for rel in CONTRACT + ["REPORT.md", "GAP_LIST.md"]:
            assert (d / rel).is_file(), f"{d.name}: missing {rel}"


def test_calibration_exams_carry_the_contract():
    folders = [d for d in (REPO / "calibration").iterdir() if d.is_dir()]
    assert len(folders) == 5, "expected exactly five calibration exams"
    for d in folders:
        for rel in CONTRACT:
            assert (d / rel).is_file(), f"{d.name}: missing {rel}"


def test_impl_files_are_numpy_only():
    """Hard constraint #1 of SKILL.md: no torch/jax/tf/sklearn anywhere in impl/."""
    banned = ("torch", "jax", "tensorflow", "sklearn", "scipy")
    for impl in list(REPO.glob("examples/*/impl/*.py")) + list(REPO.glob("calibration/*/impl/*.py")):
        text = impl.read_text()
        for lib in banned:
            assert lib not in text, f"{impl}: forbidden dependency '{lib}'"
