"""Which API keys exist, and writing them to the repo's gitignored .env.

A value never leaves this module: present() answers with booleans, and nothing
here echoes, logs, or passes a value to a subprocess, so it cannot appear in
`ps`. The .env line semantics deliberately match scripts/setup-mac.sh — keep
comments and unrelated keys, replace a key in place rather than appending a
duplicate, mode 0600, write atomically. That shell implementation cannot be
imported, so this is a cross-language duplication of a small invariant set; if
one changes, change both.
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
    """A boolean per known key. Never the values themselves."""
    lines = _lines()
    found: dict[str, bool] = {}
    for name in KEYS:
        pattern = _assign(name)
        found[name] = any(
            _unquote(match.group(1))
            for line in lines if (match := pattern.match(line))
        )
    return found


def set_key(name: str, value: str) -> None:
    """Insert or replace one key, in place, leaving every other line alone."""
    if name not in KEYS:
        raise ValueError(f"unknown key: {name}")
    lines = _lines()
    pattern = _assign(name)
    for index, line in enumerate(lines):
        if pattern.match(line):
            lines[index] = f"{name}={value}"
            break
    else:
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
