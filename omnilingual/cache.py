"""Content-addressed JSON cache so API calls are never repeated."""

from __future__ import annotations

import hashlib
import json
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
        return json.loads(p.read_text(encoding="utf-8"))

    def put(self, namespace: str, key: str, value: dict) -> None:
        p = self._path(namespace, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)


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
