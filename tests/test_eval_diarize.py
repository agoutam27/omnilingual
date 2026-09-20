import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from eval_diarize import check_collapse, summarize  # noqa: E402

from omnilingual.diarize.base import Turn  # noqa: E402


def _turns() -> list[Turn]:
    return [Turn(0.0, 30.0, "0"), Turn(30.0, 45.0, "1"), Turn(45.0, 60.0, "0")]


def test_summarize_adds_durations_per_label():
    assert summarize(_turns()) == {"0": 45.0, "1": 15.0}
    assert summarize([]) == {}


def test_check_collapse_healthy():
    assert check_collapse(_turns(), 2) is None


def test_check_collapse_too_few():
    assert "expected 3" in (check_collapse(_turns(), 3) or "")


def test_check_collapse_single_owner():
    assert "expected 2" in (check_collapse([Turn(0.0, 60.0, "0")], 2) or "")


def test_check_collapse_full_ownership_flagged():
    turns = [Turn(0.0, 59.99, "0"), Turn(59.99, 60.0, "1")]
    assert "100%" in (check_collapse(turns, 2) or "")


def test_check_collapse_empty():
    assert check_collapse([], 2) == "no speech segments found"


def test_check_collapse_single_expected_single_found():
    assert check_collapse([Turn(0.0, 60.0, "0")], 1) is None
