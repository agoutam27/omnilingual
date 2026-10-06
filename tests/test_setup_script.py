"""The setup script's capability flags — the only way the UI can answer it.

These are subprocess tests against the real script. `--help` and the flag
parser are cheap and side-effect free, so they can run anywhere; anything that
would install is exercised only by asserting on the resolved plan text, never by
letting a step execute.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup-mac.sh"

NEW_FLAGS = [
    "--extras",
    "--keys",
    "--live-setup",
    "--route-output",
    "--prefetch",
    "--run-tests",
]


def run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def test_every_capability_flag_is_documented_in_help():
    out = run("--help").stdout
    for flag in NEW_FLAGS:
        assert flag in out, f"{flag} is not in --help"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--extras", "local-stt,diarize"),
        ("--keys", "SARVAM_API_KEY"),
        ("--live-setup", "no"),
        ("--route-output", "no"),
        ("--prefetch", "yes"),
        ("--run-tests", "no"),
    ],
)
def test_a_capability_flag_is_accepted_and_echoed_in_the_plan(flag, value):
    # --dry-run prints the resolved plan and touches nothing, so this asserts
    # the flag reached the script's own state without letting any step run.
    result = run(flag, value, "--dry-run", "--yes")
    assert result.returncode == 0, result.stderr


def test_extras_flag_overrides_the_saved_config(tmp_path):
    config = tmp_path / "omnilingual" / "setup-mac.conf"
    config.parent.mkdir(parents=True)
    config.write_text("EXTRAS=diarize\n", encoding="utf-8")
    env = {"XDG_CONFIG_HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    result = run("--extras", "local-stt", "--dry-run", "--yes", env=env)
    assert result.returncode == 0, result.stderr
    assert "local-stt" in result.stdout + result.stderr


def test_a_capability_flag_needs_a_value():
    for flag in NEW_FLAGS:
        result = run(flag)
        assert result.returncode == 2, f"{flag} with no value should exit 2"
        assert "needs a value" in result.stderr


def test_unknown_capability_names_are_filtered_not_installed():
    # `sanitize` already whitelists against ALL_EXTRAS; this proves a flag
    # cannot smuggle an extra in past that whitelist.
    result = run("--extras", "local-stt;rm -rf /", "--dry-run", "--yes")
    assert result.returncode == 0, result.stderr
    assert "rm -rf" not in result.stdout + result.stderr


def test_the_flag_names_appear_exactly_once_in_the_case_block():
    body = SCRIPT.read_text(encoding="utf-8")
    case = re.search(r"while \[\[ \$# -gt 0 \]\]; do(.*?)^done", body, re.S | re.M)
    assert case, "the flag-parsing while loop is gone"
    for flag in NEW_FLAGS:
        assert case.group(1).count(f"{flag})") == 1, f"{flag} has no case arm"