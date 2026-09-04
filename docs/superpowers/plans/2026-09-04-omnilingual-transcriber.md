# Omnilingual Transcriber Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python CLI that turns a Zoom local recording of a multilingual Indian-language meeting into a Markdown transcript with each segment in its original language followed by an English translation.

**Architecture:** Linear pipeline of independent modules: ffmpeg normalizes audio to 16 kHz mono WAV, a silence-aware chunker cuts it into ≤28 s pieces, each chunk is transcribed by Sarvam Saaras (auto-detected language), each transcript is translated to English by Sarvam Mayura, and a pure renderer emits Markdown. Every API result is cached on disk keyed by content hash so re-runs are free and crashes resume.

**Tech Stack:** Python ≥3.12, `uv` for env/deps, `httpx` (Sarvam REST), `typer` + `rich` (CLI), `ffmpeg`/`ffprobe` binaries via subprocess, `pytest` + `respx` for tests.

**Spec:** `docs/superpowers/specs/2026-09-04-omnilingual-transcriber-design.md`

## Global Constraints

- Python `>=3.12`. Package name `omnilingual`, import name `omnilingual`.
- Sarvam sync STT accepts audio **under 30 s**; chunks are `max_chunk_s = 28.0`, `min_chunk_s = 5.0`.
- STT: `POST https://api.sarvam.ai/speech-to-text`, multipart, header `api-subscription-key`, fields `model=saaras:v4`, `mode=transcribe`, `language_code=unknown`. Response fields: `transcript`, `language_code` (nullable), `language_probability` (nullable).
- MT: `POST https://api.sarvam.ai/translate`, JSON `{input, source_language_code, target_language_code, model, mode}`. `mayura:v1` input max **1000 chars**; `sarvam-translate:v1` max 2000 chars. Response: `translated_text`.
- Pricing constants: STT ₹30/hour, MT ₹20 per 10,000 chars.
- Audio normalized to 16 kHz, mono, 16-bit PCM WAV.
- Timestamps rendered as `HH:MM:SS`.
- Exit codes: 0 success, 1 fatal, 2 partial (any segment `status != "ok"`).
- Commits: conventional-commit style, end body with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- All tests run with `uv run pytest -q`. Tests needing ffmpeg skip when it is absent; live tests skip unless `SARVAM_API_KEY` is set.

---

## File Structure

```
pyproject.toml
README.md
.gitignore                       (exists)
omnilingual/__init__.py
omnilingual/config.py            Settings dataclass, env loading, ConfigError
omnilingual/models.py            Chunk, STTResult, Segment, Cost, Transcript + (de)serialization
omnilingual/cache.py             JsonCache + key builders
omnilingual/http.py              send_with_retry + error hierarchy
omnilingual/audio/__init__.py
omnilingual/audio/normalize.py   ensure_ffmpeg, probe_duration, normalize
omnilingual/audio/chunker.py     detect_silences, plan_chunks, cut_chunks, chunk_audio
omnilingual/stt/__init__.py
omnilingual/stt/base.py          STTProvider protocol
omnilingual/stt/sarvam.py        SarvamSTT
omnilingual/translate/__init__.py
omnilingual/translate/base.py    Translator protocol
omnilingual/translate/mayura.py  MayuraTranslator, split_text, language sets
omnilingual/render/__init__.py
omnilingual/render/markdown.py   render, render_english_only, fmt_ts, lang_share
omnilingual/pipeline.py          work_dir_for, prepare, estimate, run
omnilingual/cli.py               typer app
tests/conftest.py                make_wav fixture helper, ffmpeg skip marker
tests/test_config.py
tests/test_models.py
tests/test_cache.py
tests/test_http.py
tests/audio/test_normalize.py
tests/audio/test_chunker.py
tests/stt/test_sarvam.py
tests/translate/test_mayura.py
tests/render/test_markdown.py
tests/render/golden/interleaved.md
tests/render/golden/english_only.md
tests/test_pipeline.py
tests/test_cli.py
tests/test_e2e.py
tests/test_live.py
```

---

### Task 1: Project scaffold and Settings

**Files:**
- Create: `pyproject.toml`
- Create: `omnilingual/__init__.py`
- Create: `omnilingual/config.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_config.py`

**Interfaces:**
- Produces:
  ```python
  class ConfigError(Exception)
  @dataclass(frozen=True)
  class Settings:
      api_key: str | None
      base_url: str = "https://api.sarvam.ai"
      stt_model: str = "saaras:v4"
      mt_model: str = "mayura:v1"
      mt_mode: str = "formal"
      max_chunk_s: float = 28.0
      min_chunk_s: float = 5.0
      langs: tuple[str, ...] = ()
      stt_inr_per_hour: float = 30.0
      mt_inr_per_10k_chars: float = 20.0
      @property
      def mt_char_limit(self) -> int   # 1000 for mayura:v1, 2000 otherwise
      def require_key(self) -> str      # raises ConfigError if api_key is None
  def load_settings(api_key: str | None = None, env: Mapping[str, str] | None = None, **overrides) -> Settings
  ```

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "omnilingual"
version = "0.1.0"
description = "Transcribe multilingual Indian-language meeting recordings to Markdown with English translation"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "typer>=0.12",
    "rich>=13",
]

[project.scripts]
omnilingual = "omnilingual.cli:app"

[dependency-groups]
dev = [
    "pytest>=8",
    "respx>=0.21",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["omnilingual"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "live: hits the real Sarvam API; needs SARVAM_API_KEY",
]
```

- [ ] **Step 2: Create package init and empty tests package**

`omnilingual/__init__.py`:
```python
"""Omnilingual: multilingual Indian-language meeting transcriber."""

__version__ = "0.1.0"
```

`tests/__init__.py`: empty file.

- [ ] **Step 3: Write the failing test**

`tests/test_config.py`:
```python
import pytest

from omnilingual.config import ConfigError, Settings, load_settings


def test_load_settings_reads_key_from_env():
    s = load_settings(env={"SARVAM_API_KEY": "k123"})
    assert s.api_key == "k123"
    assert s.stt_model == "saaras:v4"
    assert s.mt_model == "mayura:v1"
    assert s.max_chunk_s == 28.0
    assert s.min_chunk_s == 5.0


def test_explicit_key_beats_env():
    s = load_settings(api_key="explicit", env={"SARVAM_API_KEY": "fromenv"})
    assert s.api_key == "explicit"


def test_missing_key_is_allowed_until_required():
    s = load_settings(env={})
    assert s.api_key is None
    with pytest.raises(ConfigError):
        s.require_key()


def test_require_key_returns_key():
    assert load_settings(api_key="abc", env={}).require_key() == "abc"


def test_overrides_and_langs_tuple():
    s = load_settings(env={}, langs=["hi-IN", "ta-IN"], max_chunk_s=20.0)
    assert s.langs == ("hi-IN", "ta-IN")
    assert s.max_chunk_s == 20.0


def test_mt_char_limit_depends_on_model():
    assert load_settings(env={}).mt_char_limit == 1000
    assert load_settings(env={}, mt_model="sarvam-translate:v1").mt_char_limit == 2000


def test_settings_is_frozen():
    s = load_settings(env={})
    with pytest.raises(Exception):
        s.api_key = "x"  # type: ignore[misc]
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv sync && uv run pytest tests/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'omnilingual.config'`

- [ ] **Step 5: Write minimal implementation**

`omnilingual/config.py`:
```python
"""Runtime settings for omnilingual."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace


class ConfigError(Exception):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str = "https://api.sarvam.ai"
    stt_model: str = "saaras:v4"
    mt_model: str = "mayura:v1"
    mt_mode: str = "formal"
    max_chunk_s: float = 28.0
    min_chunk_s: float = 5.0
    langs: tuple[str, ...] = ()
    stt_inr_per_hour: float = 30.0
    mt_inr_per_10k_chars: float = 20.0

    @property
    def mt_char_limit(self) -> int:
        return 1000 if self.mt_model == "mayura:v1" else 2000

    def require_key(self) -> str:
        if not self.api_key:
            raise ConfigError(
                "Sarvam API key missing. Set SARVAM_API_KEY or pass --api-key."
            )
        return self.api_key


def load_settings(
    api_key: str | None = None,
    env: Mapping[str, str] | None = None,
    **overrides,
) -> Settings:
    env = os.environ if env is None else env
    key = api_key or env.get("SARVAM_API_KEY") or None
    settings = Settings(api_key=key)
    if "langs" in overrides:
        overrides["langs"] = tuple(overrides["langs"])
    return replace(settings, **overrides)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -q`
Expected: `7 passed`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock omnilingual/__init__.py omnilingual/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: scaffold project and add Settings

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Data model

**Files:**
- Create: `omnilingual/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Produces:
  ```python
  SegmentStatus = Literal["ok", "stt_failed", "mt_unsupported", "mt_failed"]
  @dataclass(frozen=True) class Chunk: idx: int; start_s: float; end_s: float; wav_path: Path
      @property duration_s -> float
  @dataclass(frozen=True) class STTResult: lang: str; prob: float; text: str
  @dataclass class Segment: chunk: Chunk; lang: str; prob: float; text: str; english: str | None; status: SegmentStatus
  @dataclass class Cost: audio_seconds: float = 0.0; mt_chars: int = 0; inr_estimate: float = 0.0
  @dataclass class Transcript: source: Path; duration_s: float; segments: list[Segment]; cost: Cost
  def chunks_to_json(chunks: list[Chunk]) -> str
  def chunks_from_json(text: str) -> list[Chunk]
  ```

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:
```python
from pathlib import Path

from omnilingual.models import (
    Chunk,
    Cost,
    Segment,
    STTResult,
    Transcript,
    chunks_from_json,
    chunks_to_json,
)


def test_chunk_duration():
    c = Chunk(idx=0, start_s=1.5, end_s=4.0, wav_path=Path("/tmp/x.wav"))
    assert c.duration_s == 2.5


def test_chunks_json_roundtrip():
    chunks = [
        Chunk(0, 0.0, 27.5, Path("chunks/0000.wav")),
        Chunk(1, 27.5, 50.0, Path("chunks/0001.wav")),
    ]
    text = chunks_to_json(chunks)
    assert chunks_from_json(text) == chunks


def test_segment_defaults():
    c = Chunk(0, 0.0, 10.0, Path("a.wav"))
    seg = Segment(chunk=c, lang="hi-IN", prob=0.9, text="नमस्ते", english="Hello", status="ok")
    assert seg.english == "Hello"
    assert seg.status == "ok"


def test_transcript_holds_cost():
    t = Transcript(source=Path("m.m4a"), duration_s=60.0, segments=[], cost=Cost())
    assert t.cost.inr_estimate == 0.0
    assert STTResult("ta-IN", 0.8, "வணக்கம்").lang == "ta-IN"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'omnilingual.models'`

- [ ] **Step 3: Write minimal implementation**

`omnilingual/models.py`:
```python
"""Dataclasses passed between pipeline stages."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SegmentStatus = Literal["ok", "stt_failed", "mt_unsupported", "mt_failed"]


@dataclass(frozen=True)
class Chunk:
    idx: int
    start_s: float
    end_s: float
    wav_path: Path

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class STTResult:
    lang: str
    prob: float
    text: str


@dataclass
class Segment:
    chunk: Chunk
    lang: str
    prob: float
    text: str
    english: str | None
    status: SegmentStatus


@dataclass
class Cost:
    audio_seconds: float = 0.0
    mt_chars: int = 0
    inr_estimate: float = 0.0


@dataclass
class Transcript:
    source: Path
    duration_s: float
    segments: list[Segment] = field(default_factory=list)
    cost: Cost = field(default_factory=Cost)


def chunks_to_json(chunks: list[Chunk]) -> str:
    return json.dumps(
        [
            {"idx": c.idx, "start_s": c.start_s, "end_s": c.end_s, "wav_path": str(c.wav_path)}
            for c in chunks
        ],
        indent=2,
    )


def chunks_from_json(text: str) -> list[Chunk]:
    return [
        Chunk(idx=d["idx"], start_s=d["start_s"], end_s=d["end_s"], wav_path=Path(d["wav_path"]))
        for d in json.loads(text)
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add omnilingual/models.py tests/test_models.py
git commit -m "feat: add pipeline data model

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: JSON cache

**Files:**
- Create: `omnilingual/cache.py`
- Create: `tests/test_cache.py`

**Interfaces:**
- Produces:
  ```python
  class JsonCache:
      def __init__(self, root: Path) -> None
      def get(self, namespace: str, key: str) -> dict | None
      def put(self, namespace: str, key: str, value: dict) -> None
  def stt_key(wav_bytes: bytes, model: str, mode: str) -> str
  def mt_key(text: str, src_lang: str, tgt_lang: str, model: str) -> str
  ```
  Files land at `<root>/<namespace>/<key>.json`. Keys are 64-char sha256 hex.

- [ ] **Step 1: Write the failing test**

`tests/test_cache.py`:
```python
from omnilingual.cache import JsonCache, mt_key, stt_key


def test_miss_then_hit(tmp_path):
    cache = JsonCache(tmp_path)
    assert cache.get("stt", "abc") is None
    cache.put("stt", "abc", {"text": "hi"})
    assert cache.get("stt", "abc") == {"text": "hi"}
    assert (tmp_path / "stt" / "abc.json").exists()


def test_namespaces_are_isolated(tmp_path):
    cache = JsonCache(tmp_path)
    cache.put("stt", "k", {"a": 1})
    assert cache.get("mt", "k") is None


def test_stt_key_changes_with_model_and_mode():
    b = b"\x00\x01"
    k1 = stt_key(b, "saaras:v4", "transcribe")
    assert len(k1) == 64
    assert k1 != stt_key(b, "saaras:v3", "transcribe")
    assert k1 != stt_key(b, "saaras:v4", "translate")
    assert k1 == stt_key(b, "saaras:v4", "transcribe")


def test_mt_key_changes_with_text_and_langs():
    k = mt_key("नमस्ते", "hi-IN", "en-IN", "mayura:v1")
    assert k != mt_key("नमस्ते!", "hi-IN", "en-IN", "mayura:v1")
    assert k != mt_key("नमस्ते", "mr-IN", "en-IN", "mayura:v1")
    assert k != mt_key("नमस्ते", "hi-IN", "en-IN", "sarvam-translate:v1")


def test_put_overwrites(tmp_path):
    cache = JsonCache(tmp_path)
    cache.put("stt", "k", {"v": 1})
    cache.put("stt", "k", {"v": 2})
    assert cache.get("stt", "k") == {"v": 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cache.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'omnilingual.cache'`

- [ ] **Step 3: Write minimal implementation**

`omnilingual/cache.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cache.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add omnilingual/cache.py tests/test_cache.py
git commit -m "feat: add content-addressed JSON cache

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: HTTP retry and error hierarchy

**Files:**
- Create: `omnilingual/http.py`
- Create: `tests/test_http.py`

**Interfaces:**
- Produces:
  ```python
  class SarvamError(Exception)          # base; carries .status and .body
  class AuthError(SarvamError)          # 401, 403
  class QuotaError(SarvamError)         # 402
  class TransientError(SarvamError)     # 429/5xx/transport after retries exhausted
  def send_with_retry(
      send: Callable[[], httpx.Response],
      *,
      retries: int = 3,
      base_delay: float = 1.0,
      sleep: Callable[[float], None] = time.sleep,
      jitter: Callable[[], float] = random.random,
  ) -> httpx.Response
  ```
  Behaviour: 2xx → return. 401/403 → `AuthError` immediately. 402 → `QuotaError` immediately. 429, 5xx, or `httpx.TransportError` → sleep `base_delay * 2**attempt + jitter()` and retry, up to `retries` retries (so `retries + 1` attempts); then `TransientError`. Any other 4xx → `SarvamError` immediately.

- [ ] **Step 1: Write the failing test**

`tests/test_http.py`:
```python
import httpx
import pytest

from omnilingual.http import (
    AuthError,
    QuotaError,
    SarvamError,
    TransientError,
    send_with_retry,
)


def _resp(status: int, body: str = "{}") -> httpx.Response:
    return httpx.Response(status, text=body, request=httpx.Request("POST", "https://x"))


def _sequence(*responses):
    it = iter(responses)

    def send():
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    return send


def test_success_first_try():
    r = send_with_retry(_sequence(_resp(200)), sleep=lambda s: None, jitter=lambda: 0)
    assert r.status_code == 200


def test_retries_on_429_then_succeeds():
    slept: list[float] = []
    r = send_with_retry(
        _sequence(_resp(429), _resp(500), _resp(200)),
        base_delay=1.0,
        sleep=slept.append,
        jitter=lambda: 0.0,
    )
    assert r.status_code == 200
    assert slept == [1.0, 2.0]


def test_transport_error_is_retried():
    req = httpx.Request("POST", "https://x")
    r = send_with_retry(
        _sequence(httpx.ConnectError("boom", request=req), _resp(200)),
        sleep=lambda s: None,
        jitter=lambda: 0.0,
    )
    assert r.status_code == 200


def test_gives_up_after_retries():
    with pytest.raises(TransientError) as ei:
        send_with_retry(
            _sequence(_resp(503), _resp(503), _resp(503), _resp(503)),
            retries=3,
            sleep=lambda s: None,
            jitter=lambda: 0.0,
        )
    assert ei.value.status == 503


def test_401_raises_auth_error_without_retry():
    calls = {"n": 0}

    def send():
        calls["n"] += 1
        return _resp(401, '{"error":"bad key"}')

    with pytest.raises(AuthError) as ei:
        send_with_retry(send, sleep=lambda s: None)
    assert calls["n"] == 1
    assert "bad key" in ei.value.body


def test_403_is_auth_error():
    with pytest.raises(AuthError):
        send_with_retry(_sequence(_resp(403)), sleep=lambda s: None)


def test_402_raises_quota_error():
    with pytest.raises(QuotaError):
        send_with_retry(_sequence(_resp(402)), sleep=lambda s: None)


def test_other_4xx_is_plain_sarvam_error():
    with pytest.raises(SarvamError) as ei:
        send_with_retry(_sequence(_resp(422, "bad field")), sleep=lambda s: None)
    assert not isinstance(ei.value, (AuthError, QuotaError, TransientError))
    assert ei.value.status == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_http.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'omnilingual.http'`

- [ ] **Step 3: Write minimal implementation**

`omnilingual/http.py`:
```python
"""Retry policy and error types shared by all Sarvam API adapters."""

from __future__ import annotations

import random
import time
from collections.abc import Callable

import httpx


class SarvamError(Exception):
    def __init__(self, message: str, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class AuthError(SarvamError):
    """401/403: key missing, invalid, or not permitted."""


class QuotaError(SarvamError):
    """402: credits exhausted. Safe to resume later."""


class TransientError(SarvamError):
    """429/5xx/network failure that persisted through all retries."""


def send_with_retry(
    send: Callable[[], httpx.Response],
    *,
    retries: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> httpx.Response:
    attempt = 0
    while True:
        try:
            resp = send()
        except httpx.TransportError as exc:
            if attempt >= retries:
                raise TransientError(f"network error after {attempt + 1} attempts: {exc}") from exc
            sleep(base_delay * (2**attempt) + jitter())
            attempt += 1
            continue

        status = resp.status_code
        if 200 <= status < 300:
            return resp
        if status in (401, 403):
            raise AuthError(f"authentication failed ({status})", status, resp.text)
        if status == 402:
            raise QuotaError("Sarvam credits exhausted (402)", status, resp.text)
        if status == 429 or status >= 500:
            if attempt >= retries:
                raise TransientError(
                    f"server returned {status} after {attempt + 1} attempts", status, resp.text
                )
            sleep(base_delay * (2**attempt) + jitter())
            attempt += 1
            continue
        raise SarvamError(f"request failed ({status}): {resp.text[:200]}", status, resp.text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_http.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add omnilingual/http.py tests/test_http.py
git commit -m "feat: add HTTP retry policy and Sarvam error types

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Test audio helper and audio normalization

**Files:**
- Create: `tests/conftest.py`
- Create: `omnilingual/audio/__init__.py` (empty)
- Create: `omnilingual/audio/normalize.py`
- Create: `tests/audio/__init__.py` (empty)
- Create: `tests/audio/test_normalize.py`

**Interfaces:**
- Produces (test helper, `tests/conftest.py`):
  ```python
  def make_wav(path: Path, parts: list[tuple[str, float]], rate: int = 16000, channels: int = 1) -> Path
  # parts: [("tone", 3.0), ("silence", 0.8), ...]; writes 16-bit PCM WAV; returns path
  requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
  ```
- Produces (`omnilingual/audio/normalize.py`):
  ```python
  class FfmpegMissingError(RuntimeError)
  def ensure_ffmpeg() -> None                 # raises FfmpegMissingError if ffmpeg or ffprobe absent
  def probe_duration(path: Path) -> float     # seconds via ffprobe
  def normalize(src: Path, dst: Path) -> float  # writes 16k mono s16 WAV, returns duration_s
  ```

- [ ] **Step 1: Write the test helper**

`tests/conftest.py`:
```python
import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def make_wav(
    path: Path,
    parts: list[tuple[str, float]],
    rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Write a 16-bit PCM WAV built from ("tone"|"silence", seconds) parts."""
    frames = bytearray()
    for kind, seconds in parts:
        n = int(seconds * rate)
        for i in range(n):
            if kind == "tone":
                sample = int(8000 * math.sin(2 * math.pi * 440 * i / rate))
            else:
                sample = 0
            frames += struct.pack("<h", sample) * channels
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return path
```

`tests/audio/__init__.py` and `omnilingual/audio/__init__.py`: empty files.

- [ ] **Step 2: Write the failing test**

`tests/audio/test_normalize.py`:
```python
import wave
from pathlib import Path

import pytest

from omnilingual.audio.normalize import FfmpegMissingError, ensure_ffmpeg, normalize, probe_duration
from tests.conftest import make_wav, requires_ffmpeg


@requires_ffmpeg
def test_normalize_converts_to_16k_mono(tmp_path: Path):
    src = make_wav(tmp_path / "in.wav", [("tone", 2.0)], rate=44100, channels=2)
    dst = tmp_path / "out.wav"
    duration = normalize(src, dst)
    with wave.open(str(dst), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
    assert abs(duration - 2.0) < 0.1


@requires_ffmpeg
def test_probe_duration(tmp_path: Path):
    src = make_wav(tmp_path / "in.wav", [("tone", 1.0), ("silence", 0.5)])
    assert abs(probe_duration(src) - 1.5) < 0.05


@requires_ffmpeg
def test_ensure_ffmpeg_passes_when_installed():
    ensure_ffmpeg()


def test_ensure_ffmpeg_raises_when_missing(monkeypatch):
    monkeypatch.setattr("omnilingual.audio.normalize.shutil.which", lambda name: None)
    with pytest.raises(FfmpegMissingError) as ei:
        ensure_ffmpeg()
    assert "brew install ffmpeg" in str(ei.value)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/audio/test_normalize.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'omnilingual.audio.normalize'`

- [ ] **Step 4: Write minimal implementation**

`omnilingual/audio/normalize.py`:
```python
"""Convert any ffmpeg-readable recording to 16 kHz mono 16-bit PCM WAV."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FfmpegMissingError(RuntimeError):
    pass


def ensure_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise FfmpegMissingError(
            f"{', '.join(missing)} not found on PATH. Install with: brew install ffmpeg"
        )


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return float(out)


def normalize(src: Path, dst: Path) -> float:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )
    return probe_duration(dst)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/audio/test_normalize.py -q`
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/audio omnilingual/audio
git commit -m "feat: add ffmpeg audio normalization and WAV test helper

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Silence-aware chunker

**Files:**
- Create: `omnilingual/audio/chunker.py`
- Create: `tests/audio/test_chunker.py`

**Interfaces:**
- Consumes: `Chunk` from `omnilingual.models`; `probe_duration` from `omnilingual.audio.normalize`.
- Produces:
  ```python
  @dataclass(frozen=True) class Silence: start: float; end: float
      @property mid -> float
  def detect_silences(wav: Path, noise_db: float = -35.0, min_dur: float = 0.4) -> list[Silence]
  def plan_chunks(duration_s: float, silences: list[Silence], max_s: float, min_s: float) -> list[tuple[float, float]]
  def cut_chunks(wav: Path, spans: list[tuple[float, float]], out_dir: Path) -> list[Chunk]
  def chunk_audio(wav: Path, out_dir: Path, max_s: float, min_s: float) -> list[Chunk]
  ```
  `plan_chunks` invariants: spans tile `[0, duration_s]` exactly with no gaps or overlap; every span `<= max_s`; every span `>= min_s` unless `duration_s < min_s` (then one span). Cut points prefer the latest silence midpoint within `[cursor + min_s, cursor + max_s]`.

- [ ] **Step 1: Write the failing test**

`tests/audio/test_chunker.py`:
```python
from pathlib import Path

import pytest

from omnilingual.audio.chunker import (
    Silence,
    chunk_audio,
    cut_chunks,
    detect_silences,
    plan_chunks,
)
from tests.conftest import make_wav, requires_ffmpeg

MAX, MIN = 28.0, 5.0


def _assert_tiling(spans, duration):
    assert spans[0][0] == 0.0
    assert abs(spans[-1][1] - duration) < 1e-6
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert abs(a1 - b0) < 1e-6
    for s, e in spans:
        assert e - s <= MAX + 1e-6


def test_short_audio_is_single_chunk():
    assert plan_chunks(12.0, [], MAX, MIN) == [(0.0, 12.0)]


def test_audio_shorter_than_min_is_single_chunk():
    assert plan_chunks(2.0, [], MAX, MIN) == [(0.0, 2.0)]


def test_no_silences_hard_cuts_at_max():
    spans = plan_chunks(70.0, [], MAX, MIN)
    _assert_tiling(spans, 70.0)
    assert spans[0] == (0.0, 28.0)
    assert spans[1] == (28.0, 56.0)
    assert spans[2] == (56.0, 70.0)


def test_prefers_latest_silence_in_window():
    silences = [Silence(10.0, 10.6), Silence(24.0, 24.8), Silence(40.0, 40.4)]
    spans = plan_chunks(60.0, silences, MAX, MIN)
    _assert_tiling(spans, 60.0)
    assert spans[0] == (0.0, 24.4)  # mid of (24.0, 24.8), not 10.3, not hard 28
    assert spans[1][1] == 40.2      # mid of (40.0, 40.4) within [29.4, 52.4]


def test_silence_before_min_is_ignored():
    silences = [Silence(2.0, 2.4)]  # mid 2.2 < min_s
    spans = plan_chunks(40.0, silences, MAX, MIN)
    assert spans[0] == (0.0, 28.0)


def test_tail_never_shorter_than_min():
    # 30s with no silences: naive cut gives 28 + 2. Must instead split ~15/15.
    spans = plan_chunks(30.0, [], MAX, MIN)
    _assert_tiling(spans, 30.0)
    assert len(spans) == 2
    for s, e in spans:
        assert e - s >= MIN


def test_tail_split_uses_silence_when_available():
    spans = plan_chunks(31.0, [Silence(20.0, 20.5)], MAX, MIN)
    _assert_tiling(spans, 31.0)
    assert spans[0] == (0.0, 20.25)
    assert spans[1] == (20.25, 31.0)


@pytest.mark.parametrize("duration", [5.0, 27.9, 28.0, 28.1, 33.0, 56.0, 61.0, 5400.0])
def test_invariants_hold_for_many_durations(duration):
    silences = [Silence(t, t + 0.5) for t in range(7, int(duration), 13)]
    spans = plan_chunks(duration, silences, MAX, MIN)
    _assert_tiling(spans, duration)
    if duration >= MIN:
        for s, e in spans:
            assert e - s >= MIN - 1e-6


@requires_ffmpeg
def test_detect_silences_finds_gap(tmp_path: Path):
    wav = make_wav(tmp_path / "a.wav", [("tone", 2.0), ("silence", 1.0), ("tone", 2.0)])
    sil = detect_silences(wav)
    assert len(sil) == 1
    assert abs(sil[0].start - 2.0) < 0.1
    assert abs(sil[0].end - 3.0) < 0.1
    assert abs(sil[0].mid - 2.5) < 0.1


@requires_ffmpeg
def test_cut_chunks_writes_files_with_right_lengths(tmp_path: Path):
    wav = make_wav(tmp_path / "a.wav", [("tone", 10.0)])
    chunks = cut_chunks(wav, [(0.0, 4.0), (4.0, 10.0)], tmp_path / "chunks")
    assert [c.idx for c in chunks] == [0, 1]
    assert chunks[0].wav_path.name == "0000.wav"
    assert chunks[1].wav_path.exists()
    import wave
    with wave.open(str(chunks[1].wav_path), "rb") as w:
        assert abs(w.getnframes() / w.getframerate() - 6.0) < 0.05


@requires_ffmpeg
def test_chunk_audio_end_to_end(tmp_path: Path):
    parts = [("tone", 20.0), ("silence", 1.0), ("tone", 20.0), ("silence", 1.0), ("tone", 20.0)]
    wav = make_wav(tmp_path / "a.wav", parts)
    chunks = chunk_audio(wav, tmp_path / "chunks", MAX, MIN)
    assert len(chunks) == 3
    assert abs(chunks[0].end_s - 20.5) < 0.2
    assert all(c.duration_s <= MAX for c in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/audio/test_chunker.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'omnilingual.audio.chunker'`

- [ ] **Step 3: Write minimal implementation**

`omnilingual/audio/chunker.py`:
```python
"""Split a normalized WAV into <=max_s chunks, cutting at silences when possible."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from omnilingual.audio.normalize import probe_duration
from omnilingual.models import Chunk

_START = re.compile(r"silence_start:\s*([0-9.]+)")
_END = re.compile(r"silence_end:\s*([0-9.]+)")


@dataclass(frozen=True)
class Silence:
    start: float
    end: float

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2


def detect_silences(wav: Path, noise_db: float = -35.0, min_dur: float = 0.4) -> list[Silence]:
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats",
            "-i", str(wav),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    silences: list[Silence] = []
    start: float | None = None
    for line in proc.stderr.splitlines():
        if m := _START.search(line):
            start = float(m.group(1))
        elif (m := _END.search(line)) and start is not None:
            silences.append(Silence(start, float(m.group(1))))
            start = None
    if start is not None:  # silence ran to end of file
        silences.append(Silence(start, probe_duration(wav)))
    return silences


def _latest_mid_in(silences: list[Silence], lo: float, hi: float) -> float | None:
    mids = [s.mid for s in silences if lo <= s.mid <= hi]
    return max(mids) if mids else None


def _nearest_mid_to(silences: list[Silence], target: float, lo: float, hi: float) -> float | None:
    mids = [s.mid for s in silences if lo <= s.mid <= hi]
    return min(mids, key=lambda m: abs(m - target)) if mids else None


def plan_chunks(
    duration_s: float, silences: list[Silence], max_s: float, min_s: float
) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    while True:
        remaining = duration_s - cursor
        if remaining <= max_s:
            spans.append((cursor, duration_s))
            return spans
        if remaining < max_s + min_s:
            # A cut at max_s would leave a tail < min_s. Split the remainder in two.
            half = cursor + remaining / 2
            cut = _nearest_mid_to(silences, half, cursor + min_s, duration_s - min_s) or half
        else:
            cut = _latest_mid_in(silences, cursor + min_s, cursor + max_s) or (cursor + max_s)
        spans.append((cursor, cut))
        cursor = cut


def cut_chunks(wav: Path, spans: list[tuple[float, float]], out_dir: Path) -> list[Chunk]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []
    for idx, (start, end) in enumerate(spans):
        dst = out_dir / f"{idx:04d}.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(wav),
                "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                "-c:a", "pcm_s16le",
                str(dst),
            ],
            check=True,
            capture_output=True,
        )
        chunks.append(Chunk(idx=idx, start_s=start, end_s=end, wav_path=dst))
    return chunks


def chunk_audio(wav: Path, out_dir: Path, max_s: float, min_s: float) -> list[Chunk]:
    duration = probe_duration(wav)
    spans = plan_chunks(duration, detect_silences(wav), max_s, min_s)
    return cut_chunks(wav, spans, out_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/audio/test_chunker.py -q`
Expected: `18 passed` (8 pure tests incl. 8 parametrized cases counted individually = 15, plus 3 ffmpeg tests)

If `test_prefers_latest_silence_in_window` fails on the second assertion because of float formatting, compare with `abs(spans[1][1] - 40.2) < 1e-9`.

- [ ] **Step 5: Commit**

```bash
git add omnilingual/audio/chunker.py tests/audio/test_chunker.py
git commit -m "feat: add silence-aware audio chunker

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Sarvam STT adapter

**Files:**
- Create: `omnilingual/stt/__init__.py` (empty)
- Create: `omnilingual/stt/base.py`
- Create: `omnilingual/stt/sarvam.py`
- Create: `tests/stt/__init__.py` (empty)
- Create: `tests/stt/test_sarvam.py`

**Interfaces:**
- Consumes: `Settings` (Task 1), `STTResult` (Task 2), `send_with_retry` and errors (Task 4).
- Produces:
  ```python
  # stt/base.py
  class STTProvider(Protocol):
      model: str
      mode: str
      def transcribe(self, wav_path: Path) -> STTResult: ...
  # stt/sarvam.py
  class SarvamSTT:
      model: str; mode: str = "transcribe"
      def __init__(self, settings: Settings, client: httpx.Client | None = None, sleep: Callable[[float], None] = time.sleep) -> None
      def transcribe(self, wav_path: Path) -> STTResult
  ```
  Null `language_code` → `"unknown"`; null `language_probability` → `0.0`; `transcript` stripped.

- [ ] **Step 1: Write the failing test**

`tests/stt/test_sarvam.py`:
```python
from pathlib import Path

import httpx
import pytest
import respx

from omnilingual.config import load_settings
from omnilingual.http import AuthError
from omnilingual.stt.sarvam import SarvamSTT

STT_URL = "https://api.sarvam.ai/speech-to-text"


@pytest.fixture
def settings():
    return load_settings(api_key="test-key", env={})


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    p = tmp_path / "c.wav"
    p.write_bytes(b"RIFF....WAVEfake")
    return p


@respx.mock
def test_transcribe_sends_multipart_and_parses(settings, wav):
    route = respx.post(STT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "request_id": "r1",
                "transcript": "  नमस्ते सब लोग  ",
                "language_code": "hi-IN",
                "language_probability": 0.97,
            },
        )
    )
    result = SarvamSTT(settings, sleep=lambda s: None).transcribe(wav)

    assert result.text == "नमस्ते सब लोग"
    assert result.lang == "hi-IN"
    assert result.prob == 0.97

    req = route.calls.last.request
    assert req.headers["api-subscription-key"] == "test-key"
    ct = req.headers["content-type"]
    assert ct.startswith("multipart/form-data")
    body = req.content
    assert b'name="model"' in body and b"saaras:v4" in body
    assert b'name="mode"' in body and b"transcribe" in body
    assert b'name="language_code"' in body and b"unknown" in body
    assert b'filename="c.wav"' in body
    assert b"RIFF....WAVEfake" in body


@respx.mock
def test_null_language_fields_default(settings, wav):
    respx.post(STT_URL).mock(
        return_value=httpx.Response(
            200, json={"transcript": "hello", "language_code": None, "language_probability": None}
        )
    )
    r = SarvamSTT(settings, sleep=lambda s: None).transcribe(wav)
    assert r.lang == "unknown"
    assert r.prob == 0.0
    assert r.text == "hello"


@respx.mock
def test_retries_then_succeeds(settings, wav):
    route = respx.post(STT_URL)
    route.side_effect = [
        httpx.Response(429),
        httpx.Response(200, json={"transcript": "ok", "language_code": "en-IN", "language_probability": 1.0}),
    ]
    r = SarvamSTT(settings, sleep=lambda s: None).transcribe(wav)
    assert r.text == "ok"
    assert route.call_count == 2


@respx.mock
def test_auth_error_propagates(settings, wav):
    respx.post(STT_URL).mock(return_value=httpx.Response(401, text="nope"))
    with pytest.raises(AuthError):
        SarvamSTT(settings, sleep=lambda s: None).transcribe(wav)


def test_uses_settings_model_and_base_url(wav):
    s = load_settings(api_key="k", env={}, stt_model="saaras:v3", base_url="https://alt.example")
    with respx.mock:
        route = respx.post("https://alt.example/speech-to-text").mock(
            return_value=httpx.Response(200, json={"transcript": "x", "language_code": "ta-IN", "language_probability": 0.5})
        )
        SarvamSTT(s, sleep=lambda t: None).transcribe(wav)
        assert b"saaras:v3" in route.calls.last.request.content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/stt -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'omnilingual.stt'`

- [ ] **Step 3: Write minimal implementation**

`omnilingual/stt/base.py`:
```python
"""Protocol every speech-to-text backend satisfies."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from omnilingual.models import STTResult


class STTProvider(Protocol):
    model: str
    mode: str

    def transcribe(self, wav_path: Path) -> STTResult: ...
```

`omnilingual/stt/sarvam.py`:
```python
"""Sarvam Saaras speech-to-text over the synchronous REST endpoint (<30 s audio)."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import httpx

from omnilingual.config import Settings
from omnilingual.http import send_with_retry
from omnilingual.models import STTResult


class SarvamSTT:
    mode = "transcribe"

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0))
        self._sleep = sleep
        self.model = settings.stt_model

    def transcribe(self, wav_path: Path) -> STTResult:
        key = self._settings.require_key()
        url = f"{self._settings.base_url}/speech-to-text"
        audio = wav_path.read_bytes()

        def send() -> httpx.Response:
            return self._client.post(
                url,
                headers={"api-subscription-key": key},
                files={"file": (wav_path.name, audio, "audio/wav")},
                data={"model": self.model, "mode": self.mode, "language_code": "unknown"},
            )

        resp = send_with_retry(send, sleep=self._sleep)
        body = resp.json()
        return STTResult(
            lang=body.get("language_code") or "unknown",
            prob=float(body.get("language_probability") or 0.0),
            text=(body.get("transcript") or "").strip(),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/stt -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add omnilingual/stt tests/stt
git commit -m "feat: add Sarvam Saaras STT adapter

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Translator protocol and Mayura adapter

**Files:**
- Create: `omnilingual/translate/__init__.py` (empty)
- Create: `omnilingual/translate/base.py`
- Create: `omnilingual/translate/mayura.py`
- Create: `tests/translate/__init__.py` (empty)
- Create: `tests/translate/test_mayura.py`

**Interfaces:**
- Consumes: `Settings`, `send_with_retry`.
- Produces:
  ```python
  # translate/base.py
  class Translator(Protocol):
      model: str
      def supports(self, lang: str) -> bool: ...
      def to_english(self, text: str, src_lang: str) -> str: ...
  # translate/mayura.py
  MAYURA_LANGS: frozenset[str]           # bn gu hi kn ml mr od pa ta te (all -IN) + en-IN
  SARVAM_TRANSLATE_LANGS: frozenset[str] # MAYURA_LANGS + as brx doi kok ks mai mni ne sa sat sd ur (-IN)
  def split_text(text: str, limit: int) -> list[str]   # sentence-aware pieces each <= limit
  class MayuraTranslator:
      model: str
      def __init__(self, settings: Settings, client: httpx.Client | None = None, sleep=time.sleep) -> None
      def supports(self, lang: str) -> bool
      def to_english(self, text: str, src_lang: str) -> str
  ```
  `to_english` splits input with `split_text(text, settings.mt_char_limit)`, calls `/translate` once per piece, joins results with a single space.

- [ ] **Step 1: Write the failing test**

`tests/translate/test_mayura.py`:
```python
import json

import httpx
import pytest
import respx

from omnilingual.config import load_settings
from omnilingual.translate.mayura import (
    MAYURA_LANGS,
    SARVAM_TRANSLATE_LANGS,
    MayuraTranslator,
    split_text,
)

MT_URL = "https://api.sarvam.ai/translate"


@pytest.fixture
def settings():
    return load_settings(api_key="test-key", env={})


def test_split_text_short_is_single_piece():
    assert split_text("नमस्ते। कैसे हो?", 1000) == ["नमस्ते। कैसे हो?"]


def test_split_text_breaks_on_sentence_boundaries():
    text = "पहला वाक्य। दूसरा वाक्य। तीसरा वाक्य।"
    pieces = split_text(text, 25)
    assert pieces == ["पहला वाक्य। दूसरा वाक्य।", "तीसरा वाक्य।"] or all(len(p) <= 25 for p in pieces)
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")


def test_split_text_hard_splits_when_no_boundary():
    text = "x" * 2500
    pieces = split_text(text, 1000)
    assert [len(p) for p in pieces] == [1000, 1000, 500]


def test_split_text_handles_latin_punctuation():
    text = "First sentence. Second one? Third!"
    pieces = split_text(text, 20)
    assert all(len(p) <= 20 for p in pieces)
    assert " ".join(pieces) == text


def test_language_sets():
    assert {"hi-IN", "ta-IN", "en-IN"} <= MAYURA_LANGS
    assert "ur-IN" not in MAYURA_LANGS
    assert "ur-IN" in SARVAM_TRANSLATE_LANGS
    assert MAYURA_LANGS < SARVAM_TRANSLATE_LANGS


def test_supports_depends_on_model(settings):
    assert MayuraTranslator(settings).supports("hi-IN")
    assert not MayuraTranslator(settings).supports("ur-IN")
    assert not MayuraTranslator(settings).supports("unknown")
    s2 = load_settings(api_key="k", env={}, mt_model="sarvam-translate:v1")
    assert MayuraTranslator(s2).supports("ur-IN")


@respx.mock
def test_to_english_sends_json_and_parses(settings):
    route = respx.post(MT_URL).mock(
        return_value=httpx.Response(
            200, json={"request_id": "r", "translated_text": "Hello everyone", "source_language_code": "hi-IN"}
        )
    )
    out = MayuraTranslator(settings, sleep=lambda s: None).to_english("नमस्ते सब लोग", "hi-IN")
    assert out == "Hello everyone"
    req = route.calls.last.request
    assert req.headers["api-subscription-key"] == "test-key"
    assert json.loads(req.content) == {
        "input": "नमस्ते सब लोग",
        "source_language_code": "hi-IN",
        "target_language_code": "en-IN",
        "model": "mayura:v1",
        "mode": "formal",
    }


@respx.mock
def test_to_english_splits_long_text_and_joins(settings):
    route = respx.post(MT_URL)
    route.side_effect = lambda request: httpx.Response(
        200, json={"translated_text": "EN(" + json.loads(request.content)["input"][:3] + ")"}
    )
    text = ("क" * 600 + "। ") * 3  # ~1806 chars, three sentences
    out = MayuraTranslator(settings, sleep=lambda s: None).to_english(text, "hi-IN")
    assert route.call_count >= 2
    assert out.count("EN(") == route.call_count
    for call in route.calls:
        assert len(json.loads(call.request.content)["input"]) <= 1000


@respx.mock
def test_retries_on_5xx(settings):
    route = respx.post(MT_URL)
    route.side_effect = [httpx.Response(502), httpx.Response(200, json={"translated_text": "ok"})]
    assert MayuraTranslator(settings, sleep=lambda s: None).to_english("x", "ta-IN") == "ok"
    assert route.call_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/translate -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'omnilingual.translate'`

- [ ] **Step 3: Write minimal implementation**

`omnilingual/translate/base.py`:
```python
"""Protocol every translation backend satisfies."""

from __future__ import annotations

from typing import Protocol


class Translator(Protocol):
    model: str

    def supports(self, lang: str) -> bool: ...

    def to_english(self, text: str, src_lang: str) -> str: ...
```

`omnilingual/translate/mayura.py`:
```python
"""Sarvam text translation (mayura:v1 or sarvam-translate:v1) to English."""

from __future__ import annotations

import re
import time
from collections.abc import Callable

import httpx

from omnilingual.config import Settings
from omnilingual.http import send_with_retry

MAYURA_LANGS: frozenset[str] = frozenset(
    {"bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN"}
)
SARVAM_TRANSLATE_LANGS: frozenset[str] = MAYURA_LANGS | frozenset(
    {"as-IN", "brx-IN", "doi-IN", "kok-IN", "ks-IN", "mai-IN", "mni-IN", "ne-IN", "sa-IN", "sat-IN", "sd-IN", "ur-IN"}
)

# Split after a sentence terminator (Devanagari danda, Latin . ? !, or newline) followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[।.?!\n])\s+")


def split_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        if not sentence:
            continue
        while len(sentence) > limit:  # single sentence longer than limit: hard split
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:limit])
            sentence = sentence[limit:]
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= limit:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


class MayuraTranslator:
    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0))
        self._sleep = sleep
        self.model = settings.mt_model

    def supports(self, lang: str) -> bool:
        langs = MAYURA_LANGS if self.model == "mayura:v1" else SARVAM_TRANSLATE_LANGS
        return lang in langs

    def _translate_piece(self, piece: str, src_lang: str) -> str:
        key = self._settings.require_key()
        url = f"{self._settings.base_url}/translate"
        payload = {
            "input": piece,
            "source_language_code": src_lang,
            "target_language_code": "en-IN",
            "model": self.model,
            "mode": self._settings.mt_mode,
        }

        def send() -> httpx.Response:
            return self._client.post(url, headers={"api-subscription-key": key}, json=payload)

        resp = send_with_retry(send, sleep=self._sleep)
        return (resp.json().get("translated_text") or "").strip()

    def to_english(self, text: str, src_lang: str) -> str:
        pieces = split_text(text, self._settings.mt_char_limit)
        return " ".join(self._translate_piece(p, src_lang) for p in pieces if p.strip())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/translate -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add omnilingual/translate tests/translate
git commit -m "feat: add Translator protocol and Mayura adapter with sentence-aware splitting

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Markdown renderer

**Files:**
- Create: `omnilingual/render/__init__.py` (empty)
- Create: `omnilingual/render/markdown.py`
- Create: `tests/render/__init__.py` (empty)
- Create: `tests/render/test_markdown.py`
- Create: `tests/render/golden/interleaved.md`
- Create: `tests/render/golden/english_only.md`

**Interfaces:**
- Consumes: `Transcript`, `Segment`, `Chunk`, `Cost`.
- Produces:
  ```python
  def fmt_ts(seconds: float) -> str                        # "HH:MM:SS", floor
  def lang_share(segments: list[Segment]) -> list[tuple[str, int]]  # (lang, pct by audio duration), desc, pct ints summing ≈100
  def render(t: Transcript) -> str
  def render_english_only(t: Transcript) -> str
  ```

- [ ] **Step 1: Write golden files**

`tests/render/golden/interleaved.md`:
```markdown
# Meeting transcript — standup.m4a

Duration 00:01:40 · 4 segments · Languages: hi-IN 50%, en-IN 25%, ta-IN 25%
Estimated cost: ₹1.23

## Transcript

**[00:00:00 → 00:00:25] hi-IN**
हम आज payment dashboard के बारे में बात करेंगे।
> We will talk about the payment dashboard today.

**[00:00:25 → 00:00:50] en-IN**
Okay, let's start with the refund numbers.

**[00:00:50 → 00:01:15] ta-IN** _(translation unavailable: language not supported by translator)_
வணக்கம் எல்லோருக்கும்.

**[00:01:15 → 00:01:40] hi-IN** _(transcription failed)_
[transcription failed]
```

`tests/render/golden/english_only.md`:
```markdown
# Meeting transcript — standup.m4a (English)

We will talk about the payment dashboard today.

Okay, let's start with the refund numbers.

[ta-IN, untranslated]

[transcription failed]
```

Both golden files end with exactly one trailing newline.

- [ ] **Step 2: Write the failing test**

`tests/render/test_markdown.py`:
```python
from pathlib import Path

from omnilingual.models import Chunk, Cost, Segment, Transcript
from omnilingual.render.markdown import fmt_ts, lang_share, render, render_english_only

GOLDEN = Path(__file__).parent / "golden"


def _chunk(i: int) -> Chunk:
    return Chunk(idx=i, start_s=i * 25.0, end_s=(i + 1) * 25.0, wav_path=Path(f"{i:04d}.wav"))


def sample_transcript() -> Transcript:
    segs = [
        Segment(_chunk(0), "hi-IN", 0.97, "हम आज payment dashboard के बारे में बात करेंगे।",
                "We will talk about the payment dashboard today.", "ok"),
        Segment(_chunk(1), "en-IN", 0.99, "Okay, let's start with the refund numbers.", None, "ok"),
        Segment(_chunk(2), "ta-IN", 0.91, "வணக்கம் எல்லோருக்கும்.", None, "mt_unsupported"),
        Segment(_chunk(3), "hi-IN", 0.0, "[transcription failed]", None, "stt_failed"),
    ]
    return Transcript(source=Path("/rec/standup.m4a"), duration_s=100.0, segments=segs,
                      cost=Cost(audio_seconds=100.0, mt_chars=47, inr_estimate=1.234))


def test_fmt_ts():
    assert fmt_ts(0) == "00:00:00"
    assert fmt_ts(59.9) == "00:00:59"
    assert fmt_ts(3661) == "01:01:01"


def test_lang_share_by_duration_desc():
    assert lang_share(sample_transcript().segments) == [("hi-IN", 50), ("en-IN", 25), ("ta-IN", 25)]


def test_lang_share_empty():
    assert lang_share([]) == []


def test_render_matches_golden():
    assert render(sample_transcript()) == (GOLDEN / "interleaved.md").read_text(encoding="utf-8")


def test_render_english_only_matches_golden():
    assert render_english_only(sample_transcript()) == (GOLDEN / "english_only.md").read_text(encoding="utf-8")


def test_mt_failed_note():
    t = sample_transcript()
    t.segments = [Segment(_chunk(0), "hi-IN", 0.9, "क", None, "mt_failed")]
    out = render(t)
    assert "_(translation failed)_" in out
    assert "> " not in out
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/render -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'omnilingual.render'`

- [ ] **Step 4: Write minimal implementation**

`omnilingual/render/markdown.py`:
```python
"""Render a Transcript to Markdown. Pure functions, no I/O."""

from __future__ import annotations

from collections import defaultdict

from omnilingual.models import Segment, Transcript

_NOTES = {
    "stt_failed": "transcription failed",
    "mt_unsupported": "translation unavailable: language not supported by translator",
    "mt_failed": "translation failed",
}


def fmt_ts(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def lang_share(segments: list[Segment]) -> list[tuple[str, int]]:
    if not segments:
        return []
    by_lang: dict[str, float] = defaultdict(float)
    for seg in segments:
        by_lang[seg.lang] += seg.chunk.duration_s
    total = sum(by_lang.values()) or 1.0
    ranked = sorted(by_lang.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(lang, round(100 * dur / total)) for lang, dur in ranked]


def _header(t: Transcript, suffix: str = "") -> list[str]:
    return [f"# Meeting transcript — {t.source.name}{suffix}", ""]


def render(t: Transcript) -> str:
    lines = _header(t)
    shares = ", ".join(f"{lang} {pct}%" for lang, pct in lang_share(t.segments))
    lines.append(
        f"Duration {fmt_ts(t.duration_s)} · {len(t.segments)} segments · Languages: {shares}"
    )
    lines.append(f"Estimated cost: ₹{t.cost.inr_estimate:.2f}")
    lines += ["", "## Transcript", ""]
    for seg in t.segments:
        head = f"**[{fmt_ts(seg.chunk.start_s)} → {fmt_ts(seg.chunk.end_s)}] {seg.lang}**"
        if seg.status != "ok":
            head += f" _({_NOTES[seg.status]})_"
        lines.append(head)
        lines.append(seg.text)
        if seg.status == "ok" and seg.english and seg.lang != "en-IN":
            lines.append(f"> {seg.english}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def render_english_only(t: Transcript) -> str:
    lines = _header(t, " (English)")
    for seg in t.segments:
        if seg.status == "stt_failed":
            lines.append("[transcription failed]")
        elif seg.lang == "en-IN":
            lines.append(seg.text)
        elif seg.english:
            lines.append(seg.english)
        else:
            lines.append(f"[{seg.lang}, untranslated]")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/render -q`
Expected: `6 passed`

If a golden test fails, print both strings with `repr()` and fix whichever side has the wrong whitespace. Golden files are the contract; do not weaken the comparison.

- [ ] **Step 6: Commit**

```bash
git add omnilingual/render tests/render
git commit -m "feat: add Markdown renderer with golden tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Pipeline orchestration

**Files:**
- Create: `omnilingual/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  ```python
  Progress = Callable[[int, int, Segment], None]   # (index 1-based, total, segment)
  def work_dir_for(source: Path, root: Path) -> Path          # root / sha256(source bytes)[:12]
  def prepare(source: Path, work_dir: Path, settings: Settings) -> tuple[float, list[Chunk]]
      # normalize → work_dir/normalized.wav (skipped if exists), chunk → work_dir/chunks/, persist chunks.json; reuse if present
  def estimate(duration_s: float, chunks: list[Chunk], settings: Settings, chars_per_second: float = 15.0) -> Cost
  def run(source: Path, work_dir: Path, settings: Settings, stt: STTProvider, translator: Translator, cache: JsonCache, progress: Progress | None = None) -> Transcript
  ```
  `run` semantics per spec §7: STT failure after retries (`TransientError`, `SarvamError` other than `AuthError`/`QuotaError`) → `stt_failed` segment; translator `supports()` false → `mt_unsupported`; MT failure → `mt_failed`; `AuthError`/`QuotaError` propagate immediately (cache already holds completed chunks). Detected language outside `settings.langs` (when non-empty) → warning via `logging`. `en-IN` is never translated. Cost: `audio_seconds = duration_s`, `mt_chars = sum(len(text))` for translated segments, `inr_estimate` from settings rates.

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
import logging
from pathlib import Path

import pytest

from omnilingual.cache import JsonCache
from omnilingual.config import load_settings
from omnilingual.http import QuotaError, TransientError
from omnilingual.models import STTResult
from omnilingual.pipeline import estimate, prepare, run, work_dir_for
from tests.conftest import make_wav, requires_ffmpeg


class FakeSTT:
    model = "fake-stt"
    mode = "transcribe"

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def transcribe(self, wav_path: Path) -> STTResult:
        self.calls += 1
        r = self._results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeMT:
    model = "fake-mt"

    def __init__(self, fail_on=()):
        self.calls = 0
        self._fail_on = set(fail_on)

    def supports(self, lang: str) -> bool:
        return lang in {"hi-IN", "ta-IN", "en-IN"}

    def to_english(self, text: str, src_lang: str) -> str:
        self.calls += 1
        if text in self._fail_on:
            raise TransientError("mt down", 503)
        return f"EN[{text}]"


@pytest.fixture
def settings():
    return load_settings(api_key="k", env={}, max_chunk_s=10.0, min_chunk_s=2.0, langs=["hi-IN", "en-IN"])


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    # 3 chunks: 10 + 10 + 5 seconds (tones with tiny gaps that are below silence threshold length)
    return make_wav(tmp_path / "meeting.wav", [("tone", 25.0)])


def test_work_dir_for_is_content_addressed(tmp_path: Path):
    a = tmp_path / "a.bin"; a.write_bytes(b"same")
    b = tmp_path / "b.bin"; b.write_bytes(b"same")
    c = tmp_path / "c.bin"; c.write_bytes(b"different")
    root = tmp_path / "work"
    assert work_dir_for(a, root) == work_dir_for(b, root)
    assert work_dir_for(a, root) != work_dir_for(c, root)
    assert work_dir_for(a, root).parent == root
    assert len(work_dir_for(a, root).name) == 12


@requires_ffmpeg
def test_prepare_creates_and_reuses_chunks(tmp_path: Path, settings, recording):
    wd = tmp_path / "wd"
    duration, chunks = prepare(recording, wd, settings)
    assert abs(duration - 25.0) < 0.1
    assert (wd / "normalized.wav").exists()
    assert (wd / "chunks.json").exists()
    assert len(chunks) == 3
    assert all(c.wav_path.exists() for c in chunks)
    mtime = (wd / "chunks.json").stat().st_mtime
    duration2, chunks2 = prepare(recording, wd, settings)
    assert chunks2 == chunks
    assert (wd / "chunks.json").stat().st_mtime == mtime


def test_estimate_prices_audio_and_chars(settings):
    from omnilingual.models import Chunk
    chunks = [Chunk(0, 0, 60.0, Path("x")), Chunk(1, 60.0, 120.0, Path("y"))]
    cost = estimate(120.0, chunks, settings, chars_per_second=10.0)
    assert cost.audio_seconds == 120.0
    assert cost.mt_chars == 1200
    # 120s of ₹30/hr = ₹1.00 ; 1200 chars of ₹20/10k = ₹2.40
    assert abs(cost.inr_estimate - 3.40) < 1e-6


@requires_ffmpeg
def test_run_happy_path_and_statuses(tmp_path: Path, settings, recording, caplog):
    wd = tmp_path / "wd"
    stt = FakeSTT([
        STTResult("hi-IN", 0.95, "नमस्ते"),
        STTResult("en-IN", 0.99, "hello"),
        STTResult("ta-IN", 0.90, "வணக்கம்"),   # outside settings.langs → warning, still translated
    ])
    mt = FakeMT()
    with caplog.at_level(logging.WARNING):
        t = run(recording, wd, settings, stt, mt, JsonCache(wd / "cache"))

    assert [s.lang for s in t.segments] == ["hi-IN", "en-IN", "ta-IN"]
    assert t.segments[0].english == "EN[नमस्ते]" and t.segments[0].status == "ok"
    assert t.segments[1].english is None and t.segments[1].status == "ok"
    assert t.segments[2].english == "EN[வணக்கம்]"
    assert mt.calls == 2  # en-IN skipped
    assert "ta-IN" in caplog.text
    assert t.cost.audio_seconds == pytest.approx(25.0, abs=0.1)
    assert t.cost.mt_chars == len("नमस्ते") + len("வணக்கம்")
    assert t.source == recording


@requires_ffmpeg
def test_run_second_time_uses_cache(tmp_path: Path, settings, recording):
    wd = tmp_path / "wd"
    cache = JsonCache(wd / "cache")
    stt1 = FakeSTT([STTResult("hi-IN", 0.9, "क"), STTResult("hi-IN", 0.9, "ख"), STTResult("hi-IN", 0.9, "ग")])
    mt1 = FakeMT()
    run(recording, wd, settings, stt1, mt1, cache)
    stt2 = FakeSTT([])
    mt2 = FakeMT()
    t = run(recording, wd, settings, stt2, mt2, cache)
    assert stt2.calls == 0 and mt2.calls == 0
    assert [s.text for s in t.segments] == ["क", "ख", "ग"]


@requires_ffmpeg
def test_run_marks_failed_chunks_and_continues(tmp_path: Path, settings, recording):
    wd = tmp_path / "wd"
    stt = FakeSTT([
        TransientError("stt down", 503),
        STTResult("hi-IN", 0.9, "fail-me"),
        STTResult("kn-IN", 0.8, "ಕನ್ನಡ"),   # FakeMT does not support kn-IN
    ])
    mt = FakeMT(fail_on={"fail-me"})
    t = run(recording, wd, settings, stt, mt, JsonCache(wd / "cache"))
    assert [s.status for s in t.segments] == ["stt_failed", "mt_failed", "mt_unsupported"]
    assert t.segments[0].text == "[transcription failed]"
    assert t.segments[1].english is None
    assert t.segments[2].english is None


@requires_ffmpeg
def test_run_quota_error_propagates_after_caching(tmp_path: Path, settings, recording):
    wd = tmp_path / "wd"
    cache = JsonCache(wd / "cache")
    stt = FakeSTT([STTResult("hi-IN", 0.9, "क"), QuotaError("out of credits", 402)])
    with pytest.raises(QuotaError):
        run(recording, wd, settings, stt, FakeMT(), cache)
    # resume: only the remaining two chunks hit STT
    stt2 = FakeSTT([STTResult("hi-IN", 0.9, "ख"), STTResult("hi-IN", 0.9, "ग")])
    t = run(recording, wd, settings, stt2, FakeMT(), cache)
    assert stt2.calls == 2
    assert [s.text for s in t.segments] == ["क", "ख", "ग"]


@requires_ffmpeg
def test_progress_callback(tmp_path: Path, settings, recording):
    seen = []
    stt = FakeSTT([STTResult("en-IN", 1.0, "a"), STTResult("en-IN", 1.0, "b"), STTResult("en-IN", 1.0, "c")])
    run(recording, tmp_path / "wd", settings, stt, FakeMT(), JsonCache(tmp_path / "c"),
        progress=lambda i, n, seg: seen.append((i, n, seg.text)))
    assert seen == [(1, 3, "a"), (2, 3, "b"), (3, 3, "c")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'omnilingual.pipeline'`

- [ ] **Step 3: Write minimal implementation**

`omnilingual/pipeline.py`:
```python
"""Wire normalize → chunk → STT → translate → Transcript, with caching and resumability."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

from omnilingual.audio.chunker import chunk_audio
from omnilingual.audio.normalize import normalize, probe_duration
from omnilingual.cache import JsonCache, mt_key, stt_key
from omnilingual.config import Settings
from omnilingual.http import AuthError, QuotaError, SarvamError
from omnilingual.models import Chunk, Cost, Segment, STTResult, Transcript, chunks_from_json, chunks_to_json
from omnilingual.stt.base import STTProvider
from omnilingual.translate.base import Translator

log = logging.getLogger("omnilingual")

Progress = Callable[[int, int, Segment], None]

STT_FAILED_TEXT = "[transcription failed]"


def work_dir_for(source: Path, root: Path) -> Path:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    return root / digest


def prepare(source: Path, work_dir: Path, settings: Settings) -> tuple[float, list[Chunk]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    wav = work_dir / "normalized.wav"
    manifest = work_dir / "chunks.json"

    if wav.exists():
        duration = probe_duration(wav)
    else:
        duration = normalize(source, wav)

    if manifest.exists():
        chunks = chunks_from_json(manifest.read_text(encoding="utf-8"))
        if all(c.wav_path.exists() for c in chunks):
            return duration, chunks

    chunks = chunk_audio(wav, work_dir / "chunks", settings.max_chunk_s, settings.min_chunk_s)
    manifest.write_text(chunks_to_json(chunks), encoding="utf-8")
    return duration, chunks


def _price(audio_seconds: float, mt_chars: int, settings: Settings) -> float:
    stt = audio_seconds / 3600.0 * settings.stt_inr_per_hour
    mt = mt_chars / 10_000.0 * settings.mt_inr_per_10k_chars
    return round(stt + mt, 4)


def estimate(
    duration_s: float, chunks: list[Chunk], settings: Settings, chars_per_second: float = 15.0
) -> Cost:
    audio_seconds = sum(c.duration_s for c in chunks) or duration_s
    mt_chars = int(audio_seconds * chars_per_second)
    return Cost(audio_seconds=audio_seconds, mt_chars=mt_chars, inr_estimate=_price(audio_seconds, mt_chars, settings))


def _stt_cached(chunk: Chunk, stt: STTProvider, cache: JsonCache) -> STTResult | None:
    """Return STT result, using cache. None means STT failed non-fatally."""
    wav_bytes = chunk.wav_path.read_bytes()
    key = stt_key(wav_bytes, stt.model, stt.mode)
    hit = cache.get("stt", key)
    if hit is not None:
        if hit.get("failed"):
            return None
        return STTResult(lang=hit["lang"], prob=hit["prob"], text=hit["text"])
    try:
        result = stt.transcribe(chunk.wav_path)
    except (AuthError, QuotaError):
        raise
    except SarvamError as exc:
        log.error("chunk %d STT failed: %s", chunk.idx, exc)
        cache.put("stt", key, {"failed": True, "error": str(exc)})
        return None
    cache.put("stt", key, {"lang": result.lang, "prob": result.prob, "text": result.text})
    return result


def _mt_cached(text: str, lang: str, translator: Translator, cache: JsonCache) -> str | None:
    key = mt_key(text, lang, "en-IN", translator.model)
    hit = cache.get("mt", key)
    if hit is not None:
        return None if hit.get("failed") else hit["english"]
    try:
        english = translator.to_english(text, lang)
    except (AuthError, QuotaError):
        raise
    except SarvamError as exc:
        log.error("translation failed for %s: %s", lang, exc)
        cache.put("mt", key, {"failed": True, "error": str(exc)})
        return None
    cache.put("mt", key, {"english": english})
    return english


def run(
    source: Path,
    work_dir: Path,
    settings: Settings,
    stt: STTProvider,
    translator: Translator,
    cache: JsonCache,
    progress: Progress | None = None,
) -> Transcript:
    duration, chunks = prepare(source, work_dir, settings)
    segments: list[Segment] = []
    mt_chars = 0

    for i, chunk in enumerate(chunks, start=1):
        result = _stt_cached(chunk, stt, cache)
        if result is None:
            seg = Segment(chunk, "unknown", 0.0, STT_FAILED_TEXT, None, "stt_failed")
        else:
            if settings.langs and result.lang not in settings.langs:
                log.warning(
                    "chunk %d: detected %s (p=%.2f) outside configured languages %s",
                    chunk.idx, result.lang, result.prob, ",".join(settings.langs),
                )
            seg = Segment(chunk, result.lang, result.prob, result.text, None, "ok")
            if result.text and result.lang != "en-IN":
                if not translator.supports(result.lang):
                    seg.status = "mt_unsupported"
                else:
                    english = _mt_cached(result.text, result.lang, translator, cache)
                    if english is None:
                        seg.status = "mt_failed"
                    else:
                        seg.english = english
                        mt_chars += len(result.text)
        segments.append(seg)
        if progress:
            progress(i, len(chunks), seg)

    cost = Cost(audio_seconds=duration, mt_chars=mt_chars, inr_estimate=_price(duration, mt_chars, settings))
    return Transcript(source=source, duration_s=duration, segments=segments, cost=cost)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: `8 passed`

If `test_prepare_creates_and_reuses_chunks` yields a chunk count other than 3, check `plan_chunks(25.0, [], 10.0, 2.0)`: expected spans `(0,10),(10,20),(20,25)`. A 25 s tone has no silences.

- [ ] **Step 5: Commit**

```bash
git add omnilingual/pipeline.py tests/test_pipeline.py
git commit -m "feat: add cached, resumable transcription pipeline

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: CLI

**Files:**
- Create: `omnilingual/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_settings`, `ConfigError`, `pipeline.{work_dir_for, prepare, estimate, run}`, `SarvamSTT`, `MayuraTranslator`, `JsonCache`, `render`, `render_english_only`, `ensure_ffmpeg`, `FfmpegMissingError`, `AuthError`, `QuotaError`.
- Produces: `app: typer.Typer` with one command `transcribe`. Options per spec §9. Exit 0 / 1 / 2.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
from pathlib import Path

import pytest
from typer.testing import CliRunner

from omnilingual import cli
from omnilingual.http import QuotaError
from omnilingual.models import Chunk, Cost, Segment, Transcript

runner = CliRunner()


def _transcript(source: Path, statuses=("ok",)) -> Transcript:
    segs = [
        Segment(Chunk(i, i * 10.0, (i + 1) * 10.0, Path("x")), "hi-IN", 0.9, "क", "EN" if st == "ok" else None, st)
        for i, st in enumerate(statuses)
    ]
    return Transcript(source=source, duration_s=10.0 * len(segs), segments=segs, cost=Cost(0, 0, 0.5))


@pytest.fixture
def rec(tmp_path: Path) -> Path:
    p = tmp_path / "meeting.m4a"
    p.write_bytes(b"fake")
    return p


@pytest.fixture
def patched(monkeypatch, rec):
    """Stub out ffmpeg and network-touching pieces."""
    monkeypatch.setattr(cli, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(cli, "prepare", lambda source, wd, s: (20.0, [Chunk(0, 0, 10.0, Path("a")), Chunk(1, 10.0, 20.0, Path("b"))]))
    monkeypatch.setattr(cli, "SarvamSTT", lambda settings: object())
    monkeypatch.setattr(cli, "MayuraTranslator", lambda settings: object())
    calls = {}

    def fake_run(source, wd, settings, stt, translator, cache, progress=None):
        calls["settings"] = settings
        calls["wd"] = wd
        t = _transcript(source, calls.get("statuses", ("ok",)))
        if progress:
            for i, seg in enumerate(t.segments, 1):
                progress(i, len(t.segments), seg)
        return t

    monkeypatch.setattr(cli, "run", fake_run)
    return calls


def test_writes_markdown_and_exits_zero(rec, patched):
    result = runner.invoke(cli.app, [str(rec), "--api-key", "k"])
    assert result.exit_code == 0, result.output
    out = rec.with_suffix(".md")
    assert out.exists()
    assert "# Meeting transcript — meeting.m4a" in out.read_text(encoding="utf-8")
    assert "₹0.50" in result.output


def test_english_only_writes_second_file(rec, patched):
    result = runner.invoke(cli.app, [str(rec), "--api-key", "k", "--english-only"])
    assert result.exit_code == 0, result.output
    assert rec.with_suffix(".en.md").exists()


def test_custom_out_and_langs(rec, patched, tmp_path):
    out = tmp_path / "custom.md"
    result = runner.invoke(cli.app, [str(rec), "--api-key", "k", "--out", str(out), "--langs", "hi-IN,ta-IN"])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert patched["settings"].langs == ("hi-IN", "ta-IN")


def test_partial_success_exits_two(rec, patched):
    patched["statuses"] = ("ok", "stt_failed")
    result = runner.invoke(cli.app, [str(rec), "--api-key", "k"])
    assert result.exit_code == 2
    assert "1 segment(s) need attention" in result.output


def test_missing_key_exits_one(rec, patched, monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    result = runner.invoke(cli.app, [str(rec)])
    assert result.exit_code == 1
    assert "SARVAM_API_KEY" in result.output


def test_estimate_needs_no_key_and_makes_no_run(rec, patched, monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.setattr(cli, "run", lambda *a, **k: pytest.fail("run must not be called"))
    result = runner.invoke(cli.app, [str(rec), "--estimate"])
    assert result.exit_code == 0, result.output
    assert "2 chunks" in result.output
    assert "₹" in result.output


def test_quota_error_exits_one_with_resume_hint(rec, patched, monkeypatch):
    def boom(*a, **k):
        raise QuotaError("credits exhausted", 402)
    monkeypatch.setattr(cli, "run", boom)
    result = runner.invoke(cli.app, [str(rec), "--api-key", "k"])
    assert result.exit_code == 1
    assert "re-run" in result.output.lower()


def test_missing_ffmpeg_exits_one(rec, patched, monkeypatch):
    from omnilingual.audio.normalize import FfmpegMissingError
    def missing():
        raise FfmpegMissingError("ffmpeg not found on PATH. Install with: brew install ffmpeg")
    monkeypatch.setattr(cli, "ensure_ffmpeg", missing)
    result = runner.invoke(cli.app, [str(rec), "--api-key", "k"])
    assert result.exit_code == 1
    assert "brew install ffmpeg" in result.output


def test_missing_input_file_exits_one(tmp_path):
    result = runner.invoke(cli.app, [str(tmp_path / "nope.m4a"), "--api-key", "k"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'omnilingual.cli'`

- [ ] **Step 3: Write minimal implementation**

`omnilingual/cli.py`:
```python
"""Command-line entry point: omnilingual transcribe RECORDING [options]."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from omnilingual.audio.normalize import FfmpegMissingError, ensure_ffmpeg
from omnilingual.cache import JsonCache
from omnilingual.config import ConfigError, load_settings
from omnilingual.http import AuthError, QuotaError, SarvamError
from omnilingual.models import Segment
from omnilingual.pipeline import estimate, prepare, run, work_dir_for
from omnilingual.render.markdown import fmt_ts, render, render_english_only
from omnilingual.stt.sarvam import SarvamSTT
from omnilingual.translate.mayura import MayuraTranslator

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()
err = Console(stderr=True)


def _fail(msg: str, code: int = 1) -> None:
    console.print(f"[red]error:[/red] {msg}")
    raise typer.Exit(code)


@app.callback(invoke_without_command=True)
def transcribe(
    recording: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Zoom local recording (.m4a/.mp4/any ffmpeg input)")],
    langs: Annotated[Optional[str], typer.Option(help="Comma-separated expected languages, e.g. hi-IN,ta-IN,en-IN. Used for warnings only.")] = None,
    out: Annotated[Optional[Path], typer.Option(help="Output markdown path. Default: <recording stem>.md next to input.")] = None,
    english_only: Annotated[bool, typer.Option("--english-only", help="Also write <stem>.en.md with English text only.")] = False,
    estimate_only: Annotated[bool, typer.Option("--estimate", help="Chunk and price the recording. No API calls.")] = False,
    api_key: Annotated[Optional[str], typer.Option(help="Sarvam API key. Overrides SARVAM_API_KEY.")] = None,
    work_dir: Annotated[Optional[Path], typer.Option(help="Cache/work root. Default: <out dir>/.omnilingual")] = None,
    max_chunk_s: Annotated[float, typer.Option(help="Max chunk length in seconds (<30).")] = 28.0,
    min_chunk_s: Annotated[float, typer.Option(help="Min chunk length in seconds.")] = 5.0,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING, format="%(levelname)s %(message)s")

    out_path = out or recording.with_suffix(".md")
    work_root = work_dir or out_path.parent / ".omnilingual"
    lang_list = [s.strip() for s in langs.split(",") if s.strip()] if langs else []
    settings = load_settings(api_key=api_key, langs=lang_list, max_chunk_s=max_chunk_s, min_chunk_s=min_chunk_s)

    try:
        ensure_ffmpeg()
    except FfmpegMissingError as exc:
        _fail(str(exc))

    wd = work_dir_for(recording, work_root)

    if estimate_only:
        duration, chunks = prepare(recording, wd, settings)
        cost = estimate(duration, chunks, settings)
        console.print(f"{recording.name}: {fmt_ts(duration)} audio, {len(chunks)} chunks")
        console.print(f"Projected: STT ₹{duration / 3600 * settings.stt_inr_per_hour:.2f} + MT ~₹{cost.mt_chars / 10_000 * settings.mt_inr_per_10k_chars:.2f} = ~₹{cost.inr_estimate:.2f}")
        raise typer.Exit(0)

    try:
        settings.require_key()
    except ConfigError as exc:
        _fail(str(exc))

    def progress(i: int, n: int, seg: Segment) -> None:
        mark = "" if seg.status == "ok" else f" [yellow]{seg.status}[/yellow]"
        console.print(f"[{i}/{n}] {fmt_ts(seg.chunk.start_s)} {seg.lang} {seg.prob:.2f}{mark}")

    try:
        transcript = run(recording, wd, settings, SarvamSTT(settings), MayuraTranslator(settings), JsonCache(wd / "cache"), progress)
    except QuotaError as exc:
        _fail(f"{exc}. Progress is cached; re-run the same command to resume.")
    except AuthError as exc:
        _fail(f"{exc}. Check your Sarvam API key.")
    except SarvamError as exc:
        _fail(str(exc))

    out_path.write_text(render(transcript), encoding="utf-8")
    console.print(f"Wrote {out_path}")
    if english_only:
        en_path = out_path.with_suffix(".en.md")
        en_path.write_text(render_english_only(transcript), encoding="utf-8")
        console.print(f"Wrote {en_path}")

    bad = sum(1 for s in transcript.segments if s.status != "ok")
    console.print(f"{len(transcript.segments)} segments · estimated cost ₹{transcript.cost.inr_estimate:.2f}")
    if bad:
        console.print(f"[yellow]{bad} segment(s) need attention (see notes in transcript).[/yellow]")
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
```

Note: a single-command Typer app registered via `@app.callback(invoke_without_command=True)` means the user runs `omnilingual RECORDING`, and the spec's `omnilingual transcribe RECORDING` form is not needed. If you prefer the explicit subcommand, use `@app.command()` instead and pass `["transcribe", str(rec), ...]` in every `runner.invoke` call in the tests. Either is acceptable; pick one and keep tests consistent. The `[project.scripts]` entry `omnilingual = "omnilingual.cli:app"` works for both.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add omnilingual/cli.py tests/test_cli.py
git commit -m "feat: add omnilingual CLI with estimate, english-only, and exit codes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: End-to-end test with mocked Sarvam API

**Files:**
- Create: `tests/test_e2e.py`

**Interfaces:**
- Consumes: the real `SarvamSTT`, `MayuraTranslator`, `pipeline.run`, `render`. Only HTTP is mocked.

- [ ] **Step 1: Write the failing test**

`tests/test_e2e.py`:
```python
import json
from pathlib import Path

import httpx
import respx

from omnilingual.cache import JsonCache
from omnilingual.config import load_settings
from omnilingual.pipeline import run, work_dir_for
from omnilingual.render.markdown import render, render_english_only
from omnilingual.stt.sarvam import SarvamSTT
from omnilingual.translate.mayura import MayuraTranslator
from tests.conftest import make_wav, requires_ffmpeg

STT_URL = "https://api.sarvam.ai/speech-to-text"
MT_URL = "https://api.sarvam.ai/translate"

STT_SCRIPT = [
    {"transcript": "हम आज payment dashboard पर बात करेंगे।", "language_code": "hi-IN", "language_probability": 0.96},
    {"transcript": "Refund numbers look fine this week.", "language_code": "en-IN", "language_probability": 0.99},
    {"transcript": "வணக்கம், நான் தொடங்குகிறேன்.", "language_code": "ta-IN", "language_probability": 0.93},
]


def _stt_side_effect():
    it = iter(STT_SCRIPT)
    return lambda request: httpx.Response(200, json=next(it))


def _mt_side_effect(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(200, json={"translated_text": f"[{body['source_language_code']}→en] {body['input']}"})


@requires_ffmpeg
@respx.mock
def test_full_pipeline_three_languages_then_cached(tmp_path: Path):
    # 3 speech blocks separated by 1s silences → 3 chunks with max_chunk_s=28
    rec = make_wav(tmp_path / "meeting.wav",
                   [("tone", 20.0), ("silence", 1.0), ("tone", 20.0), ("silence", 1.0), ("tone", 20.0)])
    settings = load_settings(api_key="k", env={}, langs=["hi-IN", "ta-IN", "en-IN"])
    stt_route = respx.post(STT_URL).mock(side_effect=_stt_side_effect())
    mt_route = respx.post(MT_URL).mock(side_effect=_mt_side_effect)

    wd = work_dir_for(rec, tmp_path / ".omnilingual")
    cache = JsonCache(wd / "cache")
    stt = SarvamSTT(settings, sleep=lambda s: None)
    mt = MayuraTranslator(settings, sleep=lambda s: None)

    t = run(rec, wd, settings, stt, mt, cache)

    assert stt_route.call_count == 3
    assert mt_route.call_count == 2  # en-IN chunk not translated
    assert [s.lang for s in t.segments] == ["hi-IN", "en-IN", "ta-IN"]
    assert all(s.status == "ok" for s in t.segments)

    md = render(t)
    assert "**[00:00:00 → 00:00:20] hi-IN**" in md
    assert "> [hi-IN→en] हम आज payment dashboard पर बात करेंगे।" in md
    assert "Refund numbers look fine this week." in md
    assert md.count("> ") == 2
    assert "Languages: hi-IN 3" in md or "Languages: ta-IN 3" in md or "Languages: en-IN 3" in md

    en = render_english_only(t)
    assert "[ta-IN→en] வணக்கம், நான் தொடங்குகிறேன்." in en
    assert "[hi-IN→en]" in en and "Refund numbers" in en

    # second run: zero network calls, identical output
    t2 = run(rec, wd, settings, SarvamSTT(settings, sleep=lambda s: None), MayuraTranslator(settings, sleep=lambda s: None), cache)
    assert stt_route.call_count == 3
    assert mt_route.call_count == 2
    assert render(t2) == md
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_e2e.py -q`
Expected: `1 passed`

This test should pass on first run because every component already exists. If it fails, the failure is a real integration bug between modules; fix the module, not the test. Likely culprits: chunk boundaries landing at 20.5 instead of 20.0 (the assertion uses `fmt_ts`, which floors, so `00:00:20` still matches), or the `Languages:` line format.

- [ ] **Step 3: Run the whole suite**

Run: `uv run pytest -q`
Expected: all tests pass, none fail. `live` tests do not exist yet.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: add end-to-end pipeline test with mocked Sarvam API

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: Live smoke test and README

**Files:**
- Create: `tests/test_live.py`
- Create: `README.md`

**Interfaces:**
- Consumes: `SarvamSTT`, `MayuraTranslator`, `make_wav`.

- [ ] **Step 1: Write the live test**

`tests/test_live.py`:
```python
"""Opt-in tests that hit the real Sarvam API. Run: SARVAM_API_KEY=... uv run pytest -m live"""

import os
from pathlib import Path

import pytest

from omnilingual.config import load_settings
from omnilingual.stt.sarvam import SarvamSTT
from omnilingual.translate.mayura import MayuraTranslator
from tests.conftest import make_wav

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("SARVAM_API_KEY"), reason="SARVAM_API_KEY not set"),
]


def test_live_stt_returns_language_and_text(tmp_path: Path):
    # A pure tone has no speech; we only assert the request shape is accepted (HTTP 200) and fields parse.
    wav = make_wav(tmp_path / "tone.wav", [("tone", 3.0)])
    result = SarvamSTT(load_settings()).transcribe(wav)
    assert isinstance(result.text, str)
    assert isinstance(result.lang, str)
    assert 0.0 <= result.prob <= 1.0


def test_live_translate_hindi_to_english():
    out = MayuraTranslator(load_settings()).to_english("नमस्ते, आप कैसे हैं?", "hi-IN")
    assert out
    assert any(word in out.lower() for word in ("hello", "how", "you"))
```

- [ ] **Step 2: Run to confirm it skips without a key, then (optionally) with one**

Run: `uv run pytest tests/test_live.py -q`
Expected: `2 skipped`

Optional, costs a few paise: `SARVAM_API_KEY=<key> uv run pytest -m live -q`
Expected: `2 passed`. If STT returns 4xx for the pure tone, replace the fixture with a short real speech clip at `tests/fixtures/hi-short.wav` (record 5 s of Hindi speech, 16 kHz mono) and load it instead of `make_wav`. Note the actual status code Sarvam returns when credits run out; if it is not 402, update `send_with_retry` in `omnilingual/http.py` to map that code to `QuotaError` and add a test case in `tests/test_http.py`.

- [ ] **Step 3: Write README**

`README.md`:
```markdown
# omnilingual

Transcribe a Zoom local recording of a meeting held in several Indian languages
into a Markdown transcript, each segment in its original language followed by an
English translation. Uses Sarvam AI (Saaras speech-to-text with per-chunk language
auto-detect, Mayura translation).

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- ffmpeg: `brew install ffmpeg`
- A Sarvam AI API key: https://dashboard.sarvam.ai

## Install

```bash
uv sync
export SARVAM_API_KEY=your-key
```

## Use

In Zoom, enable **Record to this computer**. After the meeting, run:

```bash
uv run omnilingual ~/Documents/Zoom/2026-09-04\ Standup/audio_only.m4a --langs hi-IN,ta-IN,en-IN
```

Output: `audio_only.md` next to the recording.

Useful flags:

| Flag | Effect |
|---|---|
| `--estimate` | Chunk the audio and print projected cost. No API calls. |
| `--english-only` | Also write `<stem>.en.md` with only English text. |
| `--out PATH` | Custom output path. |
| `--langs a,b,c` | Expected languages; segments detected outside the set are logged as warnings. |
| `--work-dir PATH` | Where normalized audio, chunks, and the API cache live (default `.omnilingual/` beside the output). |

Exit codes: `0` success, `2` transcript written but some segments failed (see notes in the file), `1` fatal.

## Cost and resumability

Sarvam bills about ₹30 per hour of audio for speech-to-text and ₹20 per 10,000
characters for translation. Every chunk's API result is cached on disk, so
re-running the same command on the same file makes no new API calls. If credits run
out mid-way, fix credits and re-run; only the remaining chunks are sent.

## Development

```bash
uv run pytest -q                       # unit + integration (mocked HTTP)
SARVAM_API_KEY=... uv run pytest -m live   # two real API calls, costs a few paise
```

Design: `docs/superpowers/specs/2026-09-04-omnilingual-transcriber-design.md`
```

- [ ] **Step 4: Run whole suite one final time**

Run: `uv run pytest -q`
Expected: all pass; 2 skipped (live).

- [ ] **Step 5: Commit**

```bash
git add tests/test_live.py README.md
git commit -m "docs: add README and opt-in live Sarvam smoke tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage**

| Spec section | Task |
|---|---|
| §2 input any ffmpeg container, 16 kHz mono | Task 5 |
| §3 chunked ≤28 s, silence-aware, per-chunk detect | Tasks 6, 7 |
| §4/§5 module boundaries | File structure; Tasks 1–11 |
| §6 data model, work dir layout | Tasks 2, 10 (`prepare`, `work_dir_for`) |
| §7 error table: ffmpeg missing, retry, 401/403, 402 resume, stt_failed, mt_unsupported, mt_failed, out-of-set warning, Ctrl-C | Tasks 4, 5, 10, 11 (Ctrl-C is covered because cache writes happen per chunk before the next call) |
| §7 `--estimate` | Tasks 10, 11 |
| §8 output format, english-only, percentages by duration | Task 9 |
| §9 CLI flags, progress line, exit codes | Task 11 |
| §10 tests | Tasks 1–13 |
| §11 layout, deps | Task 1 (httpx replaces `sarvamai` SDK; spec updated to match) |

**Placeholder scan:** none. Every code step has full content.

**Type consistency:** `Settings` fields used in Tasks 7, 8, 10, 11 match Task 1. `STTProvider.model/mode` (Task 7) match `stt_key(…, stt.model, stt.mode)` in Task 10. `Translator.model` (Task 8) matches `mt_key(…, translator.model)`. `Segment.status` literals match `_NOTES` keys in Task 9 and statuses set in Task 10. `Progress` signature `(int, int, Segment)` matches Task 11's `progress`.
