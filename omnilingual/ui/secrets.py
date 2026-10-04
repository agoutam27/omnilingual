"""Which API keys exist, and writing them to the repo's gitignored .env.

A value never leaves this module: present() answers with booleans, and nothing
here echoes, logs, or passes a value to a subprocess, so it cannot appear in
`ps`. The .env line semantics deliberately match scripts/setup-mac.sh — keep
comments and unrelated keys, rewrite every copy of a key in place rather than
appending a duplicate or leaving a stale one, mode 0600, write atomically. That
shell implementation cannot be imported, so this is a cross-language duplication
of a small invariant set; if one changes, change both.

Two deliberate divergences from the shell, both applied uniformly across
present/set_key/clear so the three can never disagree about which lines belong
to a key:

  - `export NAME=` counts as the key's line. setup-mac.sh's pattern does not
    match it at all, and would append a second copy; treating it as the key is
    the more useful behaviour.
  - present() resolves a duplicated key to its last line, matching env_get's
    `tail -n 1`.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
from pathlib import Path

KEYS: tuple[str, ...] = ("SARVAM_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY")

REPO_ROOT = Path(__file__).resolve().parents[2]


def env_path() -> Path:
    """The .env the CLI itself reads. Overridable so tests need no repo."""
    override = os.environ.get("OMNILINGUAL_ENV_FILE")
    if override:
        return Path(override)
    return REPO_ROOT / ".env"


def _assign(name: str) -> re.Pattern[str]:
    # Tolerate `export NAME=`, surrounding whitespace, and any quoting.
    return re.compile(r"^\s*(?:export\s+)?" + re.escape(name) + r"\s*=\s*(.*?)\s*$")


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _lines() -> list[str]:
    try:
        return env_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def present() -> dict[str, bool]:
    """A boolean per known key. Never the values themselves.

    The last matching line decides, like env_get's `tail -n 1` and like the
    dotenv loader `uv run --env-file` uses, so this answers "would the run see
    this key" rather than "does some line mention it".
    """
    lines = _lines()
    found: dict[str, bool] = {}
    for name in KEYS:
        value = ""
        for line in lines:
            if match := _assign(name).match(line):
                value = _unquote(match.group(1))
        found[name] = bool(value)
    return found


def set_key(name: str, value: str) -> None:
    """Rewrite every line for this key in place, appending when there is none.

    All matches, not the first: env_set's loop has no break, so on a
    hand-duplicated key it rewrites every copy. Rewriting only the first would
    leave the second holding the previous value — a live secret surviving a
    rotation the user believes was complete.
    """
    if name not in KEYS:
        raise ValueError(f"unknown key: {name}")
    lines = _lines()
    pattern = _assign(name)
    replaced = False
    for index, line in enumerate(lines):
        if pattern.match(line):
            lines[index] = f"{name}={value}"
            replaced = True
    if not replaced:
        lines.append(f"{name}={value}")
    _write(lines)


def clear(name: str) -> None:
    """Remove a key's line. A no-op when the key is not there."""
    if name not in KEYS:
        raise ValueError(f"unknown key: {name}")
    lines = _lines()
    pattern = _assign(name)
    kept = [line for line in lines if not pattern.match(line)]
    if len(kept) != len(lines):
        _write(kept)


def _write(lines: list[str]) -> None:
    path = env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".env-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
