"""Content-addressed JSON cache so API calls are never repeated."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


class JsonCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / f"{key}.json"

    def get(self, namespace: str, key: str) -> dict | None:
        p = self._path(namespace, key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # A truncated or corrupt entry is worth one more API call, not a crash.
            return None

    def put(self, namespace: str, key: str, value: dict) -> None:
        p = self._path(namespace, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # A temp name unique per writer: live workers can put one key at once (two
        # chunks saying the same short phrase), and a shared name let the loser's
        # replace() find its file already moved.
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{key}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(value, ensure_ascii=False, indent=2))
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def _sha(*parts: bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
        h.update(b"\x00")
    return h.hexdigest()


def stt_key(wav_bytes: bytes, model: str, mode: str) -> str:
    return _sha(wav_bytes, model.encode(), mode.encode())


def mt_key(text: str, src_lang: str, tgt_lang: str, model: str) -> str:
    return _sha(text.encode("utf-8"), src_lang.encode(), tgt_lang.encode(), model.encode())
