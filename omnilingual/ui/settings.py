"""Run preferences for the UI, persisted outside the repo.

Deliberately separate from the repo's .env (see secrets.py) and from the CLI:
phase 1 does not make the CLI read this file. It lives beside setup-mac.conf
with the same permissions, so a preferences file is never group- or
world-readable.
"""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
import tomllib
from pathlib import Path

# One entry per control in the spec's parameter table. The numeric and provider
# defaults match the CLI's own, so the panel opens on the configuration the
# command line would have used; the ones the CLI derives at runtime (out, langs,
# num_speakers) carry the panel's own opening values instead.
DEFAULTS: dict[str, object] = {
    "mode": "live",
    "source": "",
    "out": "standup.md",
    "stt": "sarvam",
    "stt_model": "",
    "mt": "mayura",
    "mt_model": "",
    "diarize": False,
    "num_speakers": 3,
    "langs": [],
    "english_only": False,
    # "" is deliberate, not a missing value. cli.py computes
    # `work_dir or out_path.parent / ".omnilingual"`, so empty falls through to
    # the effective default — .omnilingual beside the output file — whereas
    # storing the literal ".omnilingual" would pin the work directory to the
    # process CWD instead. §9's table abbreviates that effective default; it is
    # not a literal for the UI to store.
    "work_dir": "",
    "device": "Omnilingual",
    "mic_only": False,
    "target_s": 8.0,
    "max_chunk_s": 28.0,
    "min_chunk_s": 5.0,
    "noise_db": -35.0,
    "stt_workers": 2,
    "max_cost": 50.0,
}


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "omnilingual"


def settings_path() -> Path:
    return config_dir() / "ui.toml"


def load() -> dict[str, object]:
    """Return the defaults merged under the stored file.

    Never raises: an unreadable or corrupt store must not stop the app from
    opening, so the user just gets defaults and can re-save over the damage.
    """
    values = dict(DEFAULTS)
    try:
        stored = tomllib.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # ValueError covers TOMLDecodeError and the UnicodeDecodeError of a
        # store that is not UTF-8 — neither is an OSError, and load() promises
        # never to raise.
        return values
    for key, value in stored.items():
        if key in DEFAULTS:  # ignore junk keys rather than importing them
            values[key] = value
    return values


def _fmt(key: str, value: object) -> str:
    if isinstance(value, bool):
        return f"{key} = {'true' if value else 'false'}"
    if isinstance(value, (int, float)):
        return f"{key} = {value!r}"
    # json.dumps, not repr: repr of a value holding both quote characters emits
    # a Python literal ('it\'s "q".md'), and TOML reads that as a *literal*
    # string where a backslash means nothing — tomllib rejects the file and
    # load() silently discards every setting. JSON escapes are always valid TOML.
    if isinstance(value, list):
        return f"{key} = {json.dumps([str(v) for v in value], ensure_ascii=False)}"
    return f"{key} = {json.dumps(str(value), ensure_ascii=False)}"


def save(values) -> dict[str, object]:
    """Validate and persist atomically. Returns the stored state.

    Validation is by rejection, not coercion: an unknown key is almost always a
    renamed control, and silently dropping it would look like the UI forgot the
    setting.
    """
    unknown = sorted(set(values) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown setting(s): {', '.join(unknown)}")
    merged = load()
    merged.update(values)

    # Before anything is written, and this is the only place it has to be: the
    # route serializes this returned dict with allow_nan=False, so a non-finite
    # float becomes a 500 *after* save() already replaced the file. load() then
    # reads the inf back and the store is permanently poisoned — every later save
    # answers 500 too, and only deleting ui.toml recovers. Rejecting here means the
    # 400 arrives with the previous store still intact.
    for key, value in merged.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number (got {value!r})")

    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, stat.S_IRWXU)

    body = "\n".join(_fmt(key, merged[key]) for key in DEFAULTS) + "\n"

    # Write a sibling, then rename: a crash mid-write must not leave a
    # truncated store that load() would silently discard.
    handle, tmp = tempfile.mkstemp(dir=str(directory), prefix="ui-", suffix=".toml")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(tmp, settings_path())
    except BaseException:
        os.unlink(tmp)
        raise
    return merged
