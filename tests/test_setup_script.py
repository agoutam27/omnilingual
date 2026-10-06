"""The setup script's capability flags — the only way the UI can answer it.

These are subprocess tests against the real script. `--help` and the flag
parser are cheap and side-effect free, so they can run anywhere; anything that
would install is exercised only by asserting on the resolved plan text, never by
letting a step execute.
"""

from __future__ import annotations

import os
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

# The six plan lines that report a resolved capability answer, in the order the
# script prints them. Whitespace inside a line is collapsed before comparison so
# that re-aligning the plan's columns does not read as a behaviour change.
PLAN_LABELS = (
    "extras enabled",
    "keys tracked in .env",
    "live capture devices",
    "system output routed",
    "pre-download weights",
    "run test suite",
)

# The first step that installs anything. `save_config` runs just above it, so
# truncating here is what lets a test observe persistence without installing.
APPLY_MARKER = "# --- Xcode command-line tools (git + swiftc)"

# Every case below passes a value that is NOT the script's built-in default
# (EXTRAS=local-stt,diarize,ui / KEYS="" / LIVE_SETUP=no / ROUTE_OUTPUT=no /
# PREFETCH=yes / RUN_TESTS=yes). Asserting a line that matches the default would
# pass just as well for a flag the script parsed and then threw away.
CAPABILITY_CASES = [
    (("--extras", "local-stt,diarize"), "extras enabled : local-stt,diarize"),
    (("--keys", "SARVAM_API_KEY"), "keys tracked in .env : SARVAM_API_KEY"),
    (("--live-setup", "yes"), "live capture devices : yes"),
    # Routing survives only if the devices are being made, so this case has to
    # raise both flags: alone, --route-output yes resolves back to `no` and the
    # assertion would hold whether or not the flag was ever read.
    (("--live-setup", "yes", "--route-output", "yes"), "system output routed : yes"),
    (("--extras", "local-stt", "--prefetch", "no"), "pre-download weights : no"),
    (("--run-tests", "no"), "run test suite : no"),
]


def run(tmp_path: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run the real script against a throwaway config home.

    XDG_CONFIG_HOME is repointed at `tmp_path` so a test never reads or writes the
    answers of whoever is running it: inheriting the developer's real config makes
    these results depend on their machine, which is how nine of eleven cases used
    to pass here and fail in CI.
    """
    base = {**os.environ, "XDG_CONFIG_HOME": str(tmp_path), "UV_OFFLINE": "1"}
    if env:
        base.update(env)
    return subprocess.run(
        ["/bin/bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=300,
        env=base,
    )


def plan(result: subprocess.CompletedProcess) -> list[str]:
    """The resolved capability lines of the plan, whitespace-collapsed."""
    found = [
        re.sub(r"\s+", " ", line.strip())
        for line in result.stderr.splitlines()
        if line.strip().startswith(PLAN_LABELS)
    ]
    assert len(found) == len(PLAN_LABELS), f"expected a full plan, got {found}"
    return found


def save_only(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    """Run the script far enough to persist its answers, and no further.

    `--dry-run` exits before `save_config`, so persistence cannot be observed that
    way, and a real run goes on to install. This runs the script's own text cut
    off just above the first install step, so its real `load_config`, resolve,
    plan and `save_config` all execute while the destructive tail is not present
    in the file at all. The copy sits in a throwaway repo holding a stub
    `pyproject.toml` — which is the only reason the script resolves a repo root
    instead of trying to clone — so the checkout and its `.env` are unreachable.
    """
    body = SCRIPT.read_text(encoding="utf-8")
    assert APPLY_MARKER in body, "the install section moved; update APPLY_MARKER"

    repo = tmp_path / "sandbox"
    (repo / "scripts").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "omnilingual-sandbox"\nversion = "0.0.0"\n'
        'requires-python = ">=3.12"\ndependencies = []\n',
        encoding="utf-8",
    )
    copy = repo / "scripts" / SCRIPT.name
    copy.write_text(body.split(APPLY_MARKER)[0] + "\nexit 0\n", encoding="utf-8")

    config_home = tmp_path / "cfg"
    config_home.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        ["/bin/bash", str(copy), *args, "--yes"],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "XDG_CONFIG_HOME": str(config_home), "UV_OFFLINE": "1"},
    )


def test_every_capability_flag_is_documented_in_help(tmp_path):
    out = run(tmp_path, "--help").stdout
    for flag in NEW_FLAGS:
        assert flag in out, f"{flag} is not in --help"


@pytest.mark.parametrize(("args", "expected_line"), CAPABILITY_CASES)
def test_a_capability_flag_reaches_the_resolved_plan(tmp_path, args, expected_line):
    # --dry-run prints the resolved plan and touches nothing, so the resolved
    # line proves the flag reached the script's own state without letting any
    # step run. Asserting the whole line, not a substring of it: "local-stt"
    # appears elsewhere in the output for unrelated reasons.
    result = run(tmp_path, *args, "--dry-run", "--yes")
    assert result.returncode == 0, result.stderr
    assert expected_line in plan(result)


def test_extras_flag_overrides_the_saved_config(tmp_path):
    config = tmp_path / "omnilingual" / "setup-mac.conf"
    config.parent.mkdir(parents=True)
    config.write_text("EXTRAS=diarize\n", encoding="utf-8")
    result = run(tmp_path, "--extras", "local-stt", "--dry-run", "--yes")
    assert result.returncode == 0, result.stderr
    assert "extras enabled : local-stt" in plan(result)
    assert "extras enabled : diarize" not in plan(result)


def test_route_output_is_forced_off_without_live_capture(tmp_path):
    # The two answers are coupled: routing the system output means nothing
    # until the devices exist, so the pair has to resolve back to `no`.
    result = run(tmp_path, "--live-setup", "no", "--route-output", "yes", "--dry-run", "--yes")
    assert result.returncode == 0, result.stderr
    assert "live capture devices : no" in plan(result)
    assert "system output routed : no" in plan(result)


def test_a_flagged_run_persists_and_the_next_run_converges(tmp_path):
    first = save_only(tmp_path, "--extras", "local-stt", "--keys", "GROQ_API_KEY", "--run-tests", "no")
    assert first.returncode == 0, first.stderr

    saved = (tmp_path / "cfg" / "omnilingual" / "setup-mac.conf").read_text(encoding="utf-8")
    assert "EXTRAS=local-stt" in saved
    assert "KEYS=GROQ_API_KEY" in saved
    assert "RUN_TESTS=no" in saved

    # Must be the config home that was written to, not tmp_path, or this reads
    # defaults and passes without ever loading anything.
    second = run(tmp_path / "cfg", "--dry-run", "--yes")
    assert second.returncode == 0, second.stderr
    # No flags this time: the answers have to come from the saved config and land
    # on the same plan, so a re-run neither re-asks nor flips the result.
    assert plan(second) == plan(first)


def test_a_capability_flag_needs_a_value(tmp_path):
    for flag in NEW_FLAGS:
        result = run(tmp_path, flag)
        assert result.returncode == 2, f"{flag} with no value should exit 2"
        assert "needs a value" in result.stderr


def test_unknown_capability_names_are_filtered_not_installed(tmp_path):
    # `sanitize` already whitelists against ALL_EXTRAS; this proves a flag
    # cannot smuggle an extra in past that whitelist. The tainted item is dropped
    # whole rather than partially kept, so nothing of it reaches the plan.
    result = run(tmp_path, "--extras", "local-stt;rm -rf /", "--dry-run", "--yes")
    assert result.returncode == 0, result.stderr
    assert "rm -rf" not in result.stdout + result.stderr
    assert "extras enabled : none" in plan(result)


def test_the_flag_names_appear_exactly_once_in_the_case_block():
    body = SCRIPT.read_text(encoding="utf-8")
    case = re.search(r"while \[\[ \$# -gt 0 \]\]; do(.*?)^done", body, re.S | re.M)
    assert case, "the flag-parsing while loop is gone"
    for flag in NEW_FLAGS:
        assert case.group(1).count(f"{flag})") == 1, f"{flag} has no case arm"


def test_the_flag_declarations_sit_above_the_parse_loop():
    # The parser sets *_SET=1; a declaration placed after the loop would reset
    # all six to 0, and every flag would parse, appear in --help, and then be
    # silently discarded. REPO_URL_SET/BRANCH_SET work only because of where
    # they sit, so the new flags have to sit beside them.
    body = SCRIPT.read_text(encoding="utf-8")
    loop = body.index("while [[ $# -gt 0 ]]; do")
    for name in ("EXTRAS_SET", "KEYS_SET", "LIVE_SETUP_SET", "ROUTE_OUTPUT_SET", "PREFETCH_SET", "RUN_TESTS_SET"):
        declaration = re.search(rf"^{name}=0$", body, re.M)
        assert declaration, f"{name} is never declared"
        assert declaration.start() < loop, f"{name} is declared after the parse loop and will be reset"
