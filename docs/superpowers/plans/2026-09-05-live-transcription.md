# Live Mic + System Audio Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `live` mode to omnilingual that listens to the microphone and system audio of a running meeting (macOS) and transcribes + translates it into a continuously growing, crash-safe Markdown file with ~10–15 s latency.

**Architecture:** One ffmpeg child reads a single macOS Aggregate Device (mic + BlackHole 2ch) and emits 16 kHz mono s16le PCM on stdout with `silencedetect` events on stderr. A capture thread drains the PCM unconditionally; a byte-clock slicer seals chunks at the first silence start after a target duration; a worker pool reuses the batch pipeline's cached STT/MT calls; an ordered appender writes segments to Markdown via an inode-stable `pwrite` header + `O_APPEND` bodies, so `tail -f` always works and Ctrl+C always leaves a valid file.

**Tech Stack:** Python 3.12+ stdlib (`subprocess`, `wave`, `array`, `threading`, `signal`, `os`), ffmpeg (avfoundation), existing deps only: httpx, typer, rich; tests: pytest, respx. No new third-party dependencies.

**Spec:** `docs/superpowers/specs/2026-09-05-live-transcription-design.md` (rev 2).

## Global Constraints

- Platform: macOS only; devices are reached through ffmpeg's avfoundation input.
- Exactly one ffmpeg audio device per `-i`. Never two inputs + `amix` (clock drift, unbounded queue, level normalization) — rejected in spec §3.
- Chunk bounds: `min 5 s` default, `max 28 s` default, hard API limit `MAX_CHUNK_LIMIT_S = 30.0` s.
- Silence detection defaults: noise `-35 dB`, min duration `0.4 s` (overridable via `--noise-db`); the silencedetect stderr lines are parsed with the exact regexes used by `omnilingual/audio/chunker.py`.
- Session time is a **byte clock**: `bytes_of_PCM / 32000` (16 kHz mono s16le = 32 000 bytes/s). Wall clock is never used for chunk timestamps.
- `audioop` is forbidden (removed in Python 3.13); use `array`/`struct` for RMS.
- No new third-party dependencies.
- The transcript file must be valid Markdown at every instant (crash or Ctrl+C); header updates must not change the inode (in-place `pwrite` of a fixed-width header block; tmp+rename is forbidden).
- Silence/gaps are never sent to the API; an energy gate marks sub-floor chunks `no_speech`.
- Budget is scarce: default cost cap `--max-cost 50` (INR); after the cap, stop API calls but keep capturing/sealing to disk.
- All live status lines go to **stderr**; stdout stays clean.
- Conventional commits (`feat:`, `test:`, `refactor:`); run tests with `uv run pytest -q`.
- `tests/test_live.py` is RESERVED for the existing opt-in real-API tests. New live-mode tests live in the files listed in File Structure.

## File Structure

| File | Responsibility |
|---|---|
| `omnilingual/pipeline/__init__.py` (moved from `omnilingual/pipeline.py`) | Batch orchestration (unchanged behavior) + extracted `transcribe_chunks()` + `run_from_chunks()` for live-session recovery + `LIVE_SESSION_KIND` constant |
| `omnilingual/audio/chunker.py` (modify) | Rename `_START`/`_END` regexes to public `SILENCE_START_RE`/`SILENCE_END_RE` so live capture reuses them verbatim |
| `omnilingual/audio/live_capture.py` (new) | avfoundation device discovery/parsing, ffmpeg command building, RMS probe, layout probe, `LiveCapture` child lifecycle, `CaptureError` |
| `omnilingual/audio/live_slicer.py` (new) | Rolling PCM buffer, byte clock, first-silence-start sealing, energy gate flag, WAV writing (stdlib `wave`), chunk manifest append. Pure w.r.t. byte clock |
| `omnilingual/render/markdown.py` (modify) | Extract `format_segment()` and `format_english_line()` so batch and live render byte-identical segment bodies |
| `omnilingual/render/live.py` (new) | Crash-safe incremental writers: `LiveMarkdownWriter` (pwrite header + O_APPEND bodies), `LiveEnglishWriter` |
| `omnilingual/pipeline/live.py` (new) | Live orchestration: `process_chunk()`, `run_live()` — threading model, worker pool, ordered appender, lag monitor, cost cap, energy-floor calibration |
| `omnilingual/cli.py` (modify) | New `live` command (incl. `--check-audio`); `transcribe --from-chunks <session-dir>` recovery flag |
| `tests/audio/test_live_capture.py` (new) | Unit tests: device parsing/resolution, command building, RMS, layout parse, `LiveCapture` with fake Popen |
| `tests/audio/test_live_slicer.py` (new) | Unit tests: sealing rule, byte clock, bounds, flush, energy flag, manifest, idx continuation |
| `tests/render/test_render_live.py` (new) | Unit tests: header padding/inode stability, crash simulations, O_EXCL guard |
| `tests/pipeline/test_process_chunk.py` (new) | Unit tests: one-chunk STT+MT classification + billing flags |
| `tests/pipeline/test_run_live.py` (new) | Integration (respx + FakeCapture): batch-equivalence golden, cache re-use, ordering under slow STT, capture death, cost cap, quota |
| `tests/test_cli_live.py` (new) | CLI tests: check-audio, live wiring, validation, `--from-chunks` paths |
| `tests/test_live_manual.py` (new) | `pytest.mark.live` hardware checklist tests (opt-in via `OMNILINGUAL_MANUAL=1`) |
| `tests/conftest.py` (modify) | Add `raw_pcm()` helper (raw 16-bit mono PCM without WAV header) |

---

### Task 1: Convert `pipeline` into a package

**Files:**
- Modify: `omnilingual/pipeline.py` → moved to `omnilingual/pipeline/__init__.py` (content unchanged)

**Interfaces:**
- Consumes: existing `omnilingual/pipeline.py`
- Produces: `omnilingual.pipeline` importable exactly as before (`run`, `prepare`, `estimate`, `work_dir_for`, `_stt_cached`, `_mt_cached`, `_price`, `STT_FAILED_TEXT`, …) so later tasks can add `omnilingual/pipeline/live.py`

- [ ] **Step 1: Move the module into a package**

```bash
mkdir omnilingual/pipeline
git mv omnilingual/pipeline.py omnilingual/pipeline/__init__.py
```

- [ ] **Step 2: Verify nothing changed behaviorally**

Run: `uv run pytest -q`
Expected: full suite PASS (same counts as before the move). If any import error appears, it means a file imported `omnilingual.pipeline` internals via a path that needs the package — fix by keeping the moved file's content identical (do not edit it).

- [ ] **Step 3: Commit**

```bash
git add omnilingual/pipeline.py omnilingual/pipeline/__init__.py
git commit -m "refactor: make pipeline a package to host the live module"
```

---

### Task 2: Extract `transcribe_chunks()` and add `run_from_chunks()`

The live session recovery path (`omnilingual transcribe --from-chunks`) needs to run the batch STT/MT loop over an existing chunk manifest without a source recording. Extract the per-chunk loop out of `run()` unchanged, then build `run_from_chunks()` on top.

**Files:**
- Modify: `omnilingual/pipeline/__init__.py`
- Test: `tests/test_pipeline_from_chunks.py`

**Interfaces:**
- Consumes: `Chunk`, `Segment`, `Cost`, `Transcript`, `STTResult` (from `omnilingual.models`); `chunks_from_json` (from `omnilingual.models`); `_read_json`, `_stt_cached`, `_mt_cached`, `_price`, `STT_FAILED_TEXT` (same file)
- Produces:
  - `LIVE_SESSION_KIND = "omnilingual-live-session"` (str constant; Task 9 writes it into `session.json`, Task 10 validates it in the CLI)
  - `transcribe_chunks(chunks: list[Chunk], duration_s: float, source: Path, settings: Settings, stt: STTProvider, translator: Translator, cache: JsonCache, progress: Progress | None = None) -> Transcript`
  - `run_from_chunks(session_dir: Path, settings: Settings, stt: STTProvider, translator: Translator, cache: JsonCache, progress: Progress | None = None) -> Transcript` (raises `ValueError` with a human message for unusable session dirs)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_from_chunks.py`:

```python
import json
from pathlib import Path

import pytest

from omnilingual.cache import JsonCache
from omnilingual.config import load_settings
from omnilingual.models import STTResult, chunks_to_json, Chunk
from omnilingual.pipeline import LIVE_SESSION_KIND, run_from_chunks

from tests.conftest import make_wav


class FakeSTT:
    model = "saaras:v4"
    mode = "transcribe"

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, wav_path: Path) -> STTResult:
        self.calls += 1
        return STTResult(lang="hi-IN", prob=0.97, text="नमस्ते सब लोग")


class FakeTranslator:
    model = "mayura:v1"

    def supports(self, lang: str) -> bool:
        return lang != "en-IN" and lang != "unknown"

    def to_english(self, text: str, src_lang: str) -> str:
        return "Hello everyone"


def make_session(tmp_path: Path) -> Path:
    sd = tmp_path / "live-20260905T153000Z"
    chunks_dir = sd / "live-chunks"
    chunks_dir.mkdir(parents=True)
    chunks = []
    for i in range(2):
        wav = chunks_dir / f"{i:04d}.wav"
        make_wav(wav, [("tone", 6.0)])
        chunks.append(Chunk(idx=i, start_s=i * 7.0, end_s=i * 7.0 + 6.0, wav_path=wav))
    (sd / "session.json").write_text(
        json.dumps({"kind": LIVE_SESSION_KIND, "version": 1, "device": "Omnilingual", "mic_only": False}),
        encoding="utf-8",
    )
    (sd / "chunks.json").write_text(chunks_to_json(chunks), encoding="utf-8")
    return sd


def test_run_from_chunks_transcribes_manifest(tmp_path):
    sd = make_session(tmp_path)
    settings = load_settings(api_key="k", env={})
    stt, translator = FakeSTT(), FakeTranslator()
    t = run_from_chunks(sd, settings, stt, translator, JsonCache(sd / "cache"))

    assert stt.calls == 2
    assert [s.status for s in t.segments] == ["ok", "ok"]
    assert all(s.english == "Hello everyone" for s in t.segments)
    assert t.duration_s == t.segments[-1].chunk.end_s
    assert t.source == Path(sd.name)


def test_run_from_chunks_second_run_pays_nothing(tmp_path):
    sd = make_session(tmp_path)
    settings = load_settings(api_key="k", env={})
    cache = JsonCache(sd / "cache")
    run_from_chunks(sd, settings, FakeSTT(), FakeTranslator(), cache)

    stt2 = FakeSTT()
    t2 = run_from_chunks(sd, settings, stt2, FakeTranslator(), cache)
    assert stt2.calls == 0  # everything answered by the session cache
    assert [s.status for s in t2.segments] == ["ok", "ok"]


def test_run_from_chunks_rejects_non_session_dir(tmp_path):
    settings = load_settings(api_key="k", env={})
    with pytest.raises(ValueError, match="not a live session dir"):
        run_from_chunks(tmp_path, settings, FakeSTT(), FakeTranslator(), JsonCache(tmp_path / "cache"))


def test_run_from_chunks_rejects_wrong_kind(tmp_path):
    sd = tmp_path / "live-x"
    sd.mkdir()
    (sd / "session.json").write_text(json.dumps({"kind": "something-else"}), encoding="utf-8")
    settings = load_settings(api_key="k", env={})
    with pytest.raises(ValueError, match="not a live session dir"):
        run_from_chunks(sd, settings, FakeSTT(), FakeTranslator(), JsonCache(sd / "cache"))


def test_run_from_chunks_rejects_missing_wav(tmp_path):
    sd = make_session(tmp_path)
    (sd / "live-chunks" / "0001.wav").unlink()
    settings = load_settings(api_key="k", env={})
    with pytest.raises(ValueError, match="missing 1 chunk wav"):
        run_from_chunks(sd, settings, FakeSTT(), FakeTranslator(), JsonCache(sd / "cache"))


def test_run_from_chunks_rejects_empty_manifest(tmp_path):
    sd = make_session(tmp_path)
    (sd / "chunks.json").write_text("[]", encoding="utf-8")
    settings = load_settings(api_key="k", env={})
    with pytest.raises(ValueError, match="no sealed chunks"):
        run_from_chunks(sd, settings, FakeSTT(), FakeTranslator(), JsonCache(sd / "cache"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline_from_chunks.py -q`
Expected: FAIL with `ImportError: cannot import name 'LIVE_SESSION_KIND' from 'omnilingual.pipeline'` (and/or `run_from_chunks`).

- [ ] **Step 3: Implement**

In `omnilingual/pipeline/__init__.py`:

3a. Add the constant right after `STT_FAILED_TEXT`:

```python
LIVE_SESSION_KIND = "omnilingual-live-session"
```

3b. Extract the per-chunk loop from `run()` into a new function placed immediately before `run()`. The body is the existing loop verbatim; `run()` delegates:

```python
def transcribe_chunks(
    chunks: list[Chunk],
    duration_s: float,
    source: Path,
    settings: Settings,
    stt: STTProvider,
    translator: Translator,
    cache: JsonCache,
    progress: Progress | None = None,
) -> Transcript:
    """The batch STT → translate loop over ready-made chunks. Shared by the file
    pipeline (run), live-session recovery (run_from_chunks), and nothing else."""
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
            if not result.text.strip():
                # A successful call that found no speech: silence, music, or crosstalk.
                # Nothing to translate and nothing to show but the note.
                seg = Segment(chunk, result.lang, result.prob, "", None, "no_speech")
            else:
                seg = Segment(chunk, result.lang, result.prob, result.text, None, "ok")
                if result.lang != "en-IN":
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

    cost = Cost(audio_seconds=duration_s, mt_chars=mt_chars, inr_estimate=_price(duration_s, mt_chars, settings))
    return Transcript(source=source, duration_s=duration_s, segments=segments, cost=cost)


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
    return transcribe_chunks(chunks, duration, source, settings, stt, translator, cache, progress)


def run_from_chunks(
    session_dir: Path,
    settings: Settings,
    stt: STTProvider,
    translator: Translator,
    cache: JsonCache,
    progress: Progress | None = None,
) -> Transcript:
    """Finish a live session's sealed chunks with the batch loop.

    The session's own chunks.json is the manifest and its cache/ dir is the
    cache namespace, so chunks already transcribed during the live run cost
    nothing here; only the chunks the live run never billed are paid for.
    """
    session = _read_json(session_dir / "session.json")
    if not isinstance(session, dict) or session.get("kind") != LIVE_SESSION_KIND:
        raise ValueError(f"not a live session dir (no {LIVE_SESSION_KIND} session.json): {session_dir}")
    manifest = session_dir / "chunks.json"
    try:
        chunks = chunks_from_json(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"unusable chunk manifest in {session_dir}: {exc}") from exc
    if not chunks:
        raise ValueError(f"live session has no sealed chunks yet: {session_dir}")
    missing = [c.wav_path for c in chunks if not c.wav_path.exists()]
    if missing:
        raise ValueError(f"live session missing {len(missing)} chunk wav(s), e.g. {missing[0]}")
    return transcribe_chunks(chunks, chunks[-1].end_s, Path(session_dir.name), settings, stt, translator, cache, progress)
```

Note: the old `run()` body is fully replaced by the two lines above; delete the loop from it. `json` and `chunks_from_json` are already imported in this file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline_from_chunks.py tests/test_pipeline.py tests/test_e2e.py -q`
Expected: PASS (new tests pass; existing pipeline/e2e tests prove `run()` is behavior-identical).

- [ ] **Step 5: Commit**

```bash
git add omnilingual/pipeline/__init__.py tests/test_pipeline_from_chunks.py
git commit -m "feat: extract transcribe_chunks and add run_from_chunks for live-session recovery"
```

---

### Task 3: Extract segment formatting in `render/markdown.py`

Live mode must render segment bodies byte-identically to batch mode. Extract the per-segment block into `format_segment()` and the English-only line into `format_english_line()`; `render()` and `render_english_only()` use them unchanged.

**Files:**
- Modify: `omnilingual/render/markdown.py`
- Test: `tests/render/test_markdown.py` (extend)

**Interfaces:**
- Consumes: `Segment` (`omnilingual.models`), `_NOTES`, `fmt_ts` (same file)
- Produces:
  - `format_segment(seg: Segment) -> list[str]` — `[head, text?, "> english?", ""]`; used by `render()` here and by `LiveMarkdownWriter.append_segment()` in Task 7
  - `format_english_line(seg: Segment) -> str | None` — `None` for `no_speech`; used by `render_english_only()` here and by `LiveEnglishWriter` in Task 7

- [ ] **Step 1: Write the failing tests**

Append to `tests/render/test_markdown.py`:

```python
from omnilingual.render.markdown import format_english_line, format_segment


def test_format_segment_ok_with_translation():
    seg = Segment(_chunk(0), "hi-IN", 0.97, "नमस्ते", "Hello", "ok")
    assert format_segment(seg) == [
        "**[00:00:00 → 00:00:25] hi-IN**",
        "नमस्ते",
        "> Hello",
        "",
    ]


def test_format_segment_ok_english_has_no_quote_line():
    seg = Segment(_chunk(0), "en-IN", 0.99, "hello", None, "ok")
    assert format_segment(seg) == ["**[00:00:00 → 00:00:25] en-IN**", "hello", ""]


def test_format_segment_no_speech_has_note_and_no_text():
    seg = Segment(_chunk(1), "hi-IN", 0.1, "", None, "no_speech")
    assert format_segment(seg) == ["**[00:00:25 → 00:00:50] hi-IN** _(no speech detected)_", ""]


def test_format_english_line_variants():
    assert format_english_line(Segment(_chunk(0), "hi-IN", 0.9, "क", "EN[क]", "ok")) == "EN[क]"
    assert format_english_line(Segment(_chunk(0), "en-IN", 0.9, "hello", None, "ok")) == "hello"
    assert format_english_line(Segment(_chunk(0), "hi-IN", 0.0, "[transcription failed]", None, "stt_failed")) == "[transcription failed]"
    assert format_english_line(Segment(_chunk(0), "ta-IN", 0.9, "வ", None, "mt_failed")) == "[ta-IN, untranslated]"
    assert format_english_line(Segment(_chunk(0), "hi-IN", 0.1, "", None, "no_speech")) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/render/test_markdown.py -q`
Expected: FAIL with `ImportError: cannot import name 'format_segment' from 'omnilingual.render.markdown'`.

- [ ] **Step 3: Implement**

In `omnilingual/render/markdown.py`, add after `lang_share()`:

```python
def format_segment(seg: Segment) -> list[str]:
    """Markdown lines for one segment, shared verbatim by the batch renderer and
    the live incremental writer. Always ends with the blank separator line."""
    head = f"**[{fmt_ts(seg.chunk.start_s)} → {fmt_ts(seg.chunk.end_s)}] {seg.lang}**"
    if seg.status != "ok":
        head += f" _({_NOTES[seg.status]})_"
    lines = [head]
    if seg.status != "no_speech":
        lines.append(seg.text)
    if seg.status == "ok" and seg.english and seg.lang != "en-IN":
        lines.append(f"> {seg.english}")
    lines.append("")
    return lines


def format_english_line(seg: Segment) -> str | None:
    """The English-only rendering of one segment, or None when it contributes
    nothing (silence). Mirrors the batch render_english_only rules exactly."""
    if seg.status == "no_speech":
        return None
    if seg.status == "stt_failed":
        return "[transcription failed]"
    if seg.lang == "en-IN":
        return seg.text
    if seg.english:
        return seg.english
    return f"[{seg.lang}, untranslated]"
```

Then replace the bodies of `render()` and `render_english_only()` with:

```python
def render(t: Transcript) -> str:
    lines = _header(t)
    shares = ", ".join(f"{lang} {pct}%" for lang, pct in lang_share(t.segments))
    lines.append(
        f"Duration {fmt_ts(t.duration_s)} · {len(t.segments)} segments · Languages: {shares}"
    )
    lines.append(f"Estimated cost: ₹{t.cost.inr_estimate:.2f}")
    lines += ["", "## Transcript", ""]
    for seg in t.segments:
        lines += format_segment(seg)
    return "\n".join(lines).rstrip("\n") + "\n"


def render_english_only(t: Transcript) -> str:
    lines = _header(t, " (English)")
    for seg in t.segments:
        line = format_english_line(seg)
        if line is None:
            continue
        lines.append(line)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/render -q`
Expected: PASS — the golden tests (`test_render_matches_golden`, `test_render_english_only_matches_golden`) prove the output is byte-identical to before.

- [ ] **Step 5: Commit**

```bash
git add omnilingual/render/markdown.py tests/render/test_markdown.py
git commit -m "refactor: extract format_segment and format_english_line for the live writer"
```

---

### Task 4: Expose chunker silence regexes

Live capture parses ffmpeg `silencedetect` stderr lines with the *same* regexes the batch chunker uses (spec §4), so promote them to public names instead of importing underscored privates across modules.

**Files:**
- Modify: `omnilingual/audio/chunker.py`

**Interfaces:**
- Consumes: existing `_START`, `_END` in `chunker.py`
- Produces: `SILENCE_START_RE` and `SILENCE_END_RE` (same compiled patterns), imported by `omnilingual/audio/live_capture.py` in Task 5

- [ ] **Step 1: Rename**

In `omnilingual/audio/chunker.py`, rename `_START` → `SILENCE_START_RE` and `_END` → `SILENCE_END_RE` at their definitions and at every use inside `detect_silences`. No other change.

- [ ] **Step 2: Run tests to verify behavior unchanged**

Run: `uv run pytest tests/audio/test_chunker.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add omnilingual/audio/chunker.py
git commit -m "refactor: expose silence-detect regexes for live capture reuse"
```

---

### Task 5: Live audio capture over a single ffmpeg input

One ffmpeg child reads the macOS Aggregate device (`mic + BlackHole`, selected by substring name) and emits 16 kHz mono `s16le` PCM on stdout while `silencedetect` events stream on stderr (spec §4 topology). Exactly one `-i` is used — dual-input `amix` is forbidden (spec §2). `LiveCapture` stays backend-agnostic (constructor takes a device *name*, never an ffmpeg-ism) so a future ScreenCaptureKit backend can implement the same `open/read/close` shape (spec §3/§5).

**Files:**
- Create: `omnilingual/audio/live_capture.py`
- Test: `tests/audio/test_live_capture.py`

**Interfaces:**
- Consumes: `ensure_ffmpeg` from `omnilingual/audio/normalize.py`; `SILENCE_START_RE`, `SILENCE_END_RE` from `omnilingual/audio/chunker.py` (Task 4)
- Produces (consumed by Tasks 6, 9, 10): `CaptureError`, `AudioDevice`, `SAMPLE_RATE = 16000`, `BYTES_PER_SECOND = 32000`, `parse_devices(text) -> list[AudioDevice]`, `resolve_device(devices, query) -> AudioDevice`, `downmix_filter(mic_only) -> str`, `build_command(device_index, mic_only, noise_db) -> list[str]`, `parse_silence_line(line) -> tuple[str, float] | None`, `rms(samples) -> float`, `dbfs(v) -> float`, `read_exact(stream, n) -> bytes`, `describe_startup_failure(device_name, stderr_tail) -> str`, `class LiveCapture` with `__init__(device_name, *, mic_only=False, noise_db=-35.0)`, `open() -> None`, `read(n) -> bytes`, `__iter__`, `close() -> None`, context-manager support

- [ ] **Step 1: Write the failing tests**

```python
import io
import math
import struct
import subprocess

import pytest

from omnilingual.audio.live_capture import (
    AudioDevice,
    CaptureError,
    LiveCapture,
    build_command,
    dbfs,
    describe_startup_failure,
    downmix_filter,
    parse_devices,
    parse_silence_line,
    read_exact,
    resolve_device,
    rms,
)

LISTING = """\
[AVFoundation indev @ 0x7f8] AVFoundation video devices:
[AVFoundation indev @ 0x7f8] [0] FaceTime HD Camera
[AVFoundation indev @ 0x7f8] AVFoundation audio devices:
[AVFoundation indev @ 0x7f8] [0] MacBook Pro Microphone
[AVFoundation indev @ 0x7f8] [1] BlackHole 2ch
[AVFoundation indev @ 0x7f8] [2] Omnilingual
"""


def test_parse_devices_reads_audio_section_only():
    devices = parse_devices(LISTING)
    assert devices == [
        AudioDevice(index=0, name="MacBook Pro Microphone"),
        AudioDevice(index=1, name="BlackHole 2ch"),
        AudioDevice(index=2, name="Omnilingual"),
    ]


def test_resolve_device_substring_case_insensitive():
    devices = parse_devices(LISTING)
    assert resolve_device(devices, "omnilingual").index == 2


def test_resolve_device_no_match_lists_available():
    devices = parse_devices(LISTING)
    with pytest.raises(CaptureError, match="Omnilingual"):
        resolve_device(devices, "nope")


def test_resolve_device_ambiguous_lists_candidates():
    devices = parse_devices(LISTING + "[AVFoundation indev @ 0x7f8] [3] Omnilingual Backup\n")
    with pytest.raises(CaptureError, match="Omnilingual Backup"):
        resolve_device(devices, "omnilingual")


def test_downmix_filter_shapes():
    assert downmix_filter(True) == "aresample=16000"
    assert (
        downmix_filter(False)
        == "pan=mono|c0=0.5*c0+0.25*c1+0.25*c2,aresample=16000"
    )


def test_build_command_uses_single_input():
    cmd = build_command(2, False, -35.0)
    assert cmd.count("-i") == 1
    assert ":2" in cmd
    joined = " ".join(cmd)
    assert "asplit=2[pcm][det]" in joined
    assert "silencedetect=noise=-35.0dB:d=0.4" in joined
    assert cmd[-2:] == ["-f", "s16le"]


def test_parse_silence_lines():
    assert parse_silence_line("[silencedetect @ x] silence_start: 12.5") == ("start", 12.5)
    assert parse_silence_line("[silencedetect @ x] silence_end: 13.1") == ("end", 13.1)
    assert parse_silence_line("frame=  100 fps=0.0") is None


def test_rms_and_dbfs():
    assert rms(b"\x00" * 3200) == 0.0
    assert dbfs(0.0) == float("-inf")
    tone = struct.pack("<4h", 1000, -1000, 1000, -1000)
    assert rms(tone) == pytest.approx(1000 / 32768)
    assert dbfs(1.0) == pytest.approx(0.0)


def test_read_exact_raises_on_truncated_stream():
    with pytest.raises(CaptureError, match="ended"):
        read_exact(io.BytesIO(b"\x00" * 10), 12)


def test_startup_failure_maps_permission_error_to_settings_hint():
    msg = describe_startup_failure("Omnilingual", "Error: operation not permitted")
    assert "Privacy" in msg and "Microphone" in msg


def test_open_raises_when_device_missing(monkeypatch):
    class Done:
        returncode = 1
        stderr = LISTING
        stdout = ""

    monkeypatch.setattr("omnilingual.audio.live_capture.ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done())
    cap = LiveCapture("nope")
    with pytest.raises(CaptureError, match="Omnilingual"):
        cap.open()


def test_close_without_open_is_safe():
    LiveCapture("Omnilingual").close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/audio/test_live_capture.py -q`
Expected: FAIL with "No module named 'omnilingual.audio.live_capture'" (or `ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
"""Live audio capture: one ffmpeg child reads a macOS Aggregate device.

Stdout carries 16 kHz mono s16le PCM; stderr carries silencedetect events.
Exactly one ``-i`` input is used (dual-input amix is forbidden).
"""

from __future__ import annotations

import array
import math
import re
import subprocess
from dataclasses import dataclass
from typing import Iterator

from .chunker import SILENCE_END_RE, SILENCE_START_RE
from .normalize import ensure_ffmpeg

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # mono s16le
_READ_BLOCK = BYTES_PER_SECOND  # 1 s per iteration
_STARTUP_GRACE_S = 0.5

_DEVICE_LINE = re.compile(r"\[(\d+)\]\s+(.+?)\s*$")


class CaptureError(RuntimeError):
    """Raised when the capture child cannot start or dies mid-session."""


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str


def parse_devices(output: str) -> list[AudioDevice]:
    """Parse `ffmpeg -f avfoundation -list_devices true -i ""` output.

    Only the audio-device section is returned; video devices are ignored.
    """
    _, _, audio = output.partition("AVFoundation audio devices:")
    devices: list[AudioDevice] = []
    for line in audio.splitlines():
        m = _DEVICE_LINE.search(line)
        if m:
            devices.append(AudioDevice(index=int(m.group(1)), name=m.group(2)))
    return devices


def resolve_device(devices: list[AudioDevice], query: str) -> AudioDevice:
    """Resolve an unambiguous case-insensitive substring device name."""
    hits = [d for d in devices if query.lower() in d.name.lower()]
    if len(hits) == 1:
        return hits[0]
    names = ", ".join(f"[{d.index}] {d.name}" for d in devices) or "(none found)"
    if not hits:
        raise CaptureError(
            f"Audio device '{query}' not found. Available: {names}. "
            "See the setup steps: create the 'Omnilingual' Aggregate Device, "
            "then run `omnilingual live --check-audio`."
        )
    dupes = ", ".join(f"[{d.index}] {d.name}" for d in hits)
    raise CaptureError(
        f"Audio device name '{query}' is ambiguous: {dupes}. "
        "Rename the Aggregate Device to exactly 'Omnilingual'."
    )


def downmix_filter(mic_only: bool) -> str:
    """Downmix to 16 kHz mono.

    The Aggregate Device holds mic (1 ch) + BlackHole (2 ch) = 3 channels,
    mixed with the voice channel dominant. Mic-only mode is already mono.
    A channel-layout mismatch fails fast inside open() with a fix-it hint.
    """
    if mic_only:
        return "aresample=16000"
    return "pan=mono|c0=0.5*c0+0.25*c1+0.25*c2,aresample=16000"


def build_command(device_index: int, mic_only: bool, noise_db: float) -> list[str]:
    chain = (
        f"asplit=2[pcm][det];[pcm]{downmix_filter(mic_only)}[out];"
        f"[det]silencedetect=noise={noise_db}dB:d=0.4"
    )
    return [
        "ffmpeg", "-hide_banner", "-nostdin",
        "-f", "avfoundation", "-thread_queue_size", "8",
        "-i", f":{device_index}",
        "-af", chain,
        "-map", "[out]", "-c:a", "pcm_s16le", "-f", "s16le", "-",
    ]


def parse_silence_line(line: str) -> tuple[str, float] | None:
    m = SILENCE_START_RE.search(line)
    if m:
        return ("start", float(m.group(1)))
    m = SILENCE_END_RE.search(line)
    if m:
        return ("end", float(m.group(1)))
    return None


def rms(samples: bytes) -> float:
    """RMS energy of s16le PCM, 0.0..1.0. Uses array/struct (audioop is banned)."""
    if not samples:
        return 0.0
    vals = array.array("h")
    vals.frombytes(samples)
    return math.sqrt(sum(v * v for v in vals) / len(vals)) / 32768


def dbfs(v: float) -> float:
    return 20 * math.log10(v) if v > 0 else float("-inf")


def read_exact(stream, n: int) -> bytes:
    """Read exactly n bytes or raise CaptureError on a truncated stream."""
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        piece = stream.read(remaining)
        if not piece:
            raise CaptureError(
                f"Audio stream ended {remaining} bytes early; "
                "the capture child likely died."
            )
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)


def describe_startup_failure(device_name: str, stderr_tail: str) -> str:
    low = stderr_tail.lower()
    if "not permitted" in low or "permission" in low:
        return (
            f"Could not open audio device '{device_name}': microphone access denied. "
            "Open System Settings ▸ Privacy & Security ▸ Microphone, enable "
            "your terminal app, then re-run."
        )
    if "no such" in low or "not found" in low or "invalid" in low:
        return (
            f"Could not open audio device '{device_name}': device not found. "
            "Create the 'Omnilingual' Aggregate Device (mic + BlackHole), "
            "then run `omnilingual live --check-audio`."
        )
    return (
        f"Could not open audio device '{device_name}': {stderr_tail.strip()[-500:]} "
        "Check the Aggregate Device channel layout (mic + 2ch BlackHole) "
        "and run `omnilingual live --check-audio`."
    )


class LiveCapture:
    """An ffmpeg avfoundation child yielding gapless 16 kHz mono s16le PCM."""

    def __init__(self, device_name: str, *, mic_only: bool = False,
                 noise_db: float = -35.0) -> None:
        self.device_name = device_name
        self.mic_only = mic_only
        self.noise_db = noise_db
        self._proc: subprocess.Popen | None = None

    def open(self) -> None:
        ensure_ffmpeg()
        listing = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            capture_output=True, text=True,
        )
        device = resolve_device(
            parse_devices((listing.stderr or "") + (listing.stdout or "")),
            self.device_name,
        )
        proc = subprocess.Popen(
            build_command(device.index, self.mic_only, self.noise_db),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
        )
        try:
            proc.wait(timeout=_STARTUP_GRACE_S)
        except subprocess.TimeoutExpired:
            self._proc = proc  # healthy: still running
            return
        tail = (proc.stderr.read() or b"").decode("utf-8", "replace")[-2000:]
        raise CaptureError(describe_startup_failure(self.device_name, tail))

    def read(self, n: int) -> bytes:
        if self._proc is None or self._proc.stdout is None:
            raise CaptureError("Capture is not open; call open() first.")
        try:
            return read_exact(self._proc.stdout, n)
        except CaptureError as exc:
            raise CaptureError(f"{exc} (device '{self.device_name}')") from exc

    def __iter__(self) -> Iterator[bytes]:
        while True:
            try:
                yield self.read(_READ_BLOCK)
            except CaptureError:
                return

    @property
    def stderr(self):
        return None if self._proc is None else self._proc.stderr

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        finally:
            for pipe in (proc.stdout, proc.stderr):
                try:
                    if pipe is not None:
                        pipe.close()
                except (BrokenPipeError, ValueError):
                    pass

    def __enter__(self) -> LiveCapture:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/audio/test_live_capture.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add omnilingual/audio/live_capture.py tests/audio/test_live_capture.py
git commit -m "feat: capture live mic+system audio over a single ffmpeg input"
```

---

### Task 6: Byte-clock rolling slicer with energy gate

`LiveSlicer` is fed raw PCM bytes plus completed silence gaps (seconds on the detector timeline, which matches the downmixed byte clock). It seals at the FIRST silence-start once the buffer reaches `target_s`, hard-cuts at `max_s`, discards gap audio (never billed, spec §4), and tags each sealed chunk with speech energy so silent chunks skip paid STT (spec §2 energy gate). Pure with respect to the byte clock: session time is always `bytes / 32000`, never wall clock. Sealed speech chunks are written as WAVs plus a rewritten manifest holding speech chunks only (absolute paths, same `chunks.json` schema the batch `chunks_from_json` parses).

**Files:**
- Create: `omnilingual/audio/live_slicer.py`
- Modify: `tests/conftest.py` (add `raw_pcm()`; `make_wav()` delegates to it, tone constants unchanged)
- Test: `tests/audio/test_live_slicer.py`

**Interfaces:**
- Consumes: `Chunk` from `omnilingual/models.py`; `chunks_to_json`, `chunks_from_json` from `omnilingual/models.py`; `BYTES_PER_SECOND` from `omnilingual/audio/live_capture.py` (Task 5); `rms` from `omnilingual/audio/live_capture.py` (Task 5)
- Produces (consumed by Tasks 8, 9): `SealedChunk` (frozen dataclass: `chunk: Chunk`, `speech: bool`), `class LiveSlicer` with `__init__(session_dir, *, target_s=8.0, max_s=28.0, min_s=5.0, energy_floor=0.004)`, `feed(data) -> list[SealedChunk]`, `note_gap(start_s, end_s) -> list[SealedChunk]`, `flush() -> list[SealedChunk]`, `manifest_path: Path`

- [ ] **Step 1: Add `raw_pcm()` to conftest and delegate `make_wav()`**

In `tests/conftest.py`, add (keeping the existing 440 Hz / amplitude-8000 tone):

```python
def raw_pcm(parts: list[tuple[str, float]], rate: int = 16000) -> bytes:
    """Mono s16le PCM for ("tone"|"silence", seconds) parts. Test helper."""
    import math
    import struct

    out = bytearray()
    for kind, seconds in parts:
        n = int(seconds * rate)
        if kind == "tone":
            for i in range(n):
                out += struct.pack(
                    "<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))
                )
        elif kind == "silence":
            out += b"\x00" * (n * 2)
        else:
            raise ValueError(f"unknown part kind: {kind}")
    return bytes(out)


def make_wav(path, parts, rate: int = 16000, channels: int = 1):
    import wave

    data = raw_pcm(parts, rate)
    if channels == 2:
        interleaved = bytearray()
        for i in range(0, len(data), 2):
            interleaved += data[i : i + 2] * 2
        data = bytes(interleaved)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(data)
    return path
```

(`make_wav` keeps its name, arguments, and tone; only the sample generation moves into `raw_pcm`.)

- [ ] **Step 2: Write the failing slicer tests**

```python
from omnilingual.audio.live_slicer import LiveSlicer
from omnilingual.models import chunks_from_json
from tests.conftest import raw_pcm


def test_seals_at_first_gap_after_target(tmp_path):
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=8.0, max_s=28.0, min_s=5.0)
    sealed = s.feed(raw_pcm([("tone", 10.0)]))
    assert sealed == []  # gap not known yet: nothing sealed
    sealed = s.note_gap(10.0, 10.8)
    assert [c.chunk.start_s for c in sealed] == [0.0]
    assert sealed[0].chunk.end_s == 10.0
    assert sealed[0].speech is True


def test_ignores_gap_before_target(tmp_path):
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=8.0, max_s=28.0, min_s=5.0)
    s.feed(raw_pcm([("tone", 4.0)]))
    assert s.note_gap(4.0, 4.5) == []
    sealed = s.feed(raw_pcm([("tone", 6.0)]))
    assert sealed == []
    sealed = s.note_gap(10.0, 10.6)
    assert [(c.chunk.start_s, c.chunk.end_s) for c in sealed] == [(0.0, 10.0)]


def test_gap_audio_discarded_from_next_chunk(tmp_path):
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=8.0, max_s=28.0, min_s=5.0)
    s.feed(raw_pcm([("tone", 10.0)]))
    s.note_gap(10.0, 11.0)
    sealed = s.feed(raw_pcm([("tone", 8.0)]))
    assert sealed == []
    sealed = s.note_gap(19.0, 19.5)
    assert [(c.chunk.start_s, c.chunk.end_s) for c in sealed] == [(11.0, 19.0)]


def test_hard_cut_at_max_without_silence(tmp_path):
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=8.0, max_s=12.0, min_s=5.0)
    sealed = s.feed(raw_pcm([("tone", 12.0)]))
    assert [(c.chunk.start_s, c.chunk.end_s) for c in sealed] == [(0.0, 12.0)]


def test_flush_seals_partial_above_min_and_discards_below(tmp_path):
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=8.0, max_s=28.0, min_s=5.0)
    s.feed(raw_pcm([("tone", 6.0)]))
    sealed = s.flush()
    assert [(c.chunk.start_s, c.chunk.end_s) for c in sealed] == [(0.0, 6.0)]
    s.feed(raw_pcm([("tone", 2.0)]))
    assert s.flush() == []


def test_quiet_chunk_tagged_no_speech_and_excluded_from_manifest(tmp_path):
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=4.0, max_s=6.0, min_s=2.0)
    sealed = s.feed(raw_pcm([("silence", 6.0)]))
    assert len(sealed) == 1 and sealed[0].speech is False
    manifest = chunks_from_json((session / "chunks.json").read_text())
    assert manifest == []
    assert not (session / "live-chunks" / "0000.wav").exists()


def test_manifest_round_trips_speech_chunks(tmp_path):
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=4.0, max_s=6.0, min_s=2.0)
    s.feed(raw_pcm([("tone", 6.0)]))
    manifest = chunks_from_json((session / "chunks.json").read_text())
    assert [(c.idx, c.start_s, c.end_s) for c in manifest] == [(0, 0.0, 6.0)]
    import wave

    with wave.open(str(manifest[0].wav_path), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert w.readframes(w.getnframes()) == raw_pcm([("tone", 6.0)])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/audio/test_live_slicer.py -q`
Expected: FAIL with "No module named 'omnilingual.audio.live_slicer'" (or `ModuleNotFoundError`).

- [ ] **Step 4: Write minimal implementation**

```python
"""Rolling byte-clock slicer for live capture.

Pure with respect to the byte clock: every timestamp is bytes / 32000,
never wall clock, so seals are deterministic under pipe buffering.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

from omnilingual.models import Chunk, chunks_to_json

from .live_capture import BYTES_PER_SECOND, rms


@dataclass(frozen=True)
class SealedChunk:
    chunk: Chunk
    speech: bool


class LiveSlicer:
    """Accumulates PCM, seals speech chunks at silence gaps or the max cut."""

    def __init__(self, session_dir: Path, *, target_s: float = 8.0,
                 max_s: float = 28.0, min_s: float = 5.0,
                 energy_floor: float = 0.004) -> None:
        if not min_s <= target_s <= max_s:
            raise ValueError("require min_s <= target_s <= max_s")
        self.session_dir = session_dir
        self.target = target_s
        self.max_s = max_s
        self.min_s = min_s
        self.energy_floor = energy_floor
        self.manifest_path = session_dir / "chunks.json"
        self._wav_dir = session_dir / "live-chunks"
        self._wav_dir.mkdir(parents=True, exist_ok=True)
        self._buf = bytearray()
        self._clock = 0.0  # session seconds consumed from the stream
        self._chunk_start = 0.0
        self._skip_until: float | None = None
        self._gaps: list[tuple[float, float]] = []
        self._next_idx = 0
        self._manifest: list[Chunk] = []
        self._write_manifest()  # chunks.json always exists from birth (possibly [])

    def _buffered_s(self) -> float:
        return len(self._buf) / BYTES_PER_SECOND

    def feed(self, data: bytes) -> list[SealedChunk]:
        """Append stream bytes, dropping any span inside a known silence gap."""
        t0 = self._clock
        t1 = t0 + len(data) / BYTES_PER_SECOND
        self._clock = t1
        if self._skip_until is not None:
            if t1 <= self._skip_until:
                return []
            cut = int((self._skip_until - t0) * BYTES_PER_SECOND)
            data = data[cut:]
            self._skip_until = None
        self._buf += data
        return self._maybe_seal()

    def note_gap(self, start_s: float, end_s: float) -> list[SealedChunk]:
        """Record a completed silence gap; seal immediately if it closes a chunk."""
        if end_s <= start_s:
            return []
        self._gaps.append((start_s, end_s))
        return self._maybe_seal()

    def _maybe_seal(self) -> list[SealedChunk]:
        out: list[SealedChunk] = []
        while True:
            buffered = self._buffered_s()
            if buffered < self.target and buffered < self.max_s:
                return out
            if buffered >= self.max_s:
                sealed = self._seal(self._chunk_start + self.max_s)
                if sealed is not None:
                    out.append(sealed)
                continue
            hit = self._first_gap_at_or_after(self._chunk_start + self.target)
            if hit is None:
                return out
            start, end = hit
            if start > self._chunk_start + buffered:
                return out
            sealed = self._seal(start)
            # Discard gap audio: drop through the gap end, keep the tail.
            drop_through = int((end - self._chunk_start) * BYTES_PER_SECOND)
            del self._buf[: max(0, drop_through)]
            self._chunk_start = end
            self._gaps = [(a, b) for a, b in self._gaps if b > end]
            if sealed is not None:
                out.append(sealed)

    def _first_gap_at_or_after(self, t: float) -> tuple[float, float] | None:
        cands = [(a, b) for a, b in self._gaps if a >= t - 1e-9]
        if not cands:
            return None
        return min(cands, key=lambda g: g[0])

    def _seal(self, end_s: float) -> SealedChunk | None:
        head_len = int((end_s - self._chunk_start) * BYTES_PER_SECOND)
        head = bytes(self._buf[:head_len])
        del self._buf[:head_len]
        start, self._chunk_start = self._chunk_start, end_s
        if (end_s - start) < self.min_s - 1e-9:
            return None
        speech = rms(head) >= self.energy_floor
        wav_path = self._wav_dir / f"{self._next_idx:04d}.wav"
        self._next_idx += 1
        if speech:
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(head)
            chunk = Chunk(idx=len(self._manifest), start_s=start,
                          end_s=end_s, wav_path=wav_path.resolve())
            self._manifest.append(chunk)
            self._write_manifest()
            return SealedChunk(chunk=chunk, speech=True)
        return SealedChunk(
            chunk=Chunk(idx=-1, start_s=start, end_s=end_s, wav_path=wav_path),
            speech=False,
        )

    def _write_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.part")
        tmp.write_text(chunks_to_json(self._manifest), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def flush(self) -> list[SealedChunk]:
        """Seal the trailing partial chunk if it reaches min_s (Ctrl+C path)."""
        if self._buffered_s() < self.min_s - 1e-9:
            self._buf.clear()
            return []
        sealed = self._seal(self._chunk_start + self._buffered_s())
        return [] if sealed is None else [sealed]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/audio/test_live_slicer.py tests/audio/test_chunker.py -q`
Expected: PASS (chunker tests prove `make_wav` still produces identical fixtures).

- [ ] **Step 6: Commit**

```bash
git add omnilingual/audio/live_slicer.py tests/audio/test_live_slicer.py tests/conftest.py
git commit -m "feat: add byte-clock rolling slicer with energy gate"
```

---

### Task 7: Crash-safe incremental Markdown writer

`tail -f` follows the inode, so the live writer must never `tmp+rename` the output file (spec §8): a fixed-width space-padded 4-line header block is updated in place with `pwrite`, and each segment body is one `O_APPEND` write plus `fsync`. After a kill the header may undercount bodies — the safe direction. Segment bodies reuse `format_segment` / `format_english_line` from Task 3, so live output is body-identical to batch (spec §8). Files are created `O_EXCL`: the writer never truncates an existing transcript (the CLI resolves `--out` collisions first, Task 10).

**Files:**
- Create: `omnilingual/render/live.py`
- Test: `tests/render/test_render_live.py`

**Interfaces:**
- Consumes: `Segment`, `Chunk`, `fmt_ts`, `lang_share`, `format_segment`, `format_english_line` from `omnilingual/render/markdown.py` (Task 3)
- Produces (consumed by Task 9): `HEADER_WIDTH = 512`, `class LiveMarkdownWriter` with `__init__(path, *, title, cost_cap)`, `append_segment(seg, cost_delta=0.0) -> None`, `finalize() -> None`, `close() -> None`, context-manager support; `class LiveEnglishWriter(LiveMarkdownWriter)` whose `append_segment` is a no-op for `no_speech` segments

- [ ] **Step 1: Write the failing tests**

```python
import os

import pytest

from omnilingual.models import Chunk, Segment
from omnilingual.render.live import (
    HEADER_WIDTH,
    LiveEnglishWriter,
    LiveMarkdownWriter,
)

CHUNK = Chunk(idx=0, start_s=4.0, end_s=13.0, wav_path="x.wav")


def _seg(status="ok"):
    return Segment(chunk=CHUNK, lang="hi-IN", prob=0.97,
                   text="Namaste", english="Hello", status=status)


def test_header_block_is_fixed_width_and_updated_in_place(tmp_path):
    out = tmp_path / "meeting.md"
    with LiveMarkdownWriter(out, title="T", cost_cap=50.0) as w:
        ino = os.stat(out).st_ino
        assert os.stat(out).st_size >= HEADER_WIDTH
        w.append_segment(_seg(), cost_delta=0.05)
        assert os.stat(out).st_ino == ino
    text = out.read_text(encoding="utf-8")
    assert "00:00:04 → 00:00:13" in text
    assert "Namaste" in text and "> Hello" in text
    assert "· growing" not in text  # finalize() dropped the marker


def test_growing_marker_present_mid_session(tmp_path):
    out = tmp_path / "meeting.md"
    w = LiveMarkdownWriter(out, title="T", cost_cap=50.0)
    w.append_segment(_seg(), cost_delta=0.05)
    text = out.read_text(encoding="utf-8")
    assert "· growing" in text
    assert "Cost ₹0.05 (cap ₹50)" in text
    w.close()


def test_crash_mid_session_leaves_valid_file(tmp_path):
    out = tmp_path / "meeting.md"
    w = LiveMarkdownWriter(out, title="T", cost_cap=50.0)
    w.append_segment(_seg(), cost_delta=0.05)
    w.append_segment(_seg(status="stt_failed"), cost_delta=0.0)
    os.close(w._fd)  # simulate kill -9: no finalize, no close
    text = out.read_text(encoding="utf-8")  # must still parse as text
    assert text.count("**[00:00:04 → 00:00:13]") == 2
    assert "transcription failed" in text


def test_existing_file_is_never_truncated(tmp_path):
    out = tmp_path / "meeting.md"
    out.write_text("precious", encoding="utf-8")
    with pytest.raises(FileExistsError):
        LiveMarkdownWriter(out, title="T", cost_cap=50.0)
    assert out.read_text(encoding="utf-8") == "precious"


def test_english_writer_skips_no_speech(tmp_path):
    out = tmp_path / "meeting.en.md"
    with LiveEnglishWriter(out, title="T", cost_cap=50.0) as w:
        w.append_segment(_seg(status="no_speech"))
        w.append_segment(_seg())
    text = out.read_text(encoding="utf-8")
    assert text.count("Namaste") == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/render/test_render_live.py -q`
Expected: FAIL with "No module named 'omnilingual.render.live'" (or `ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
"""Crash-safe incremental Markdown writer for live sessions.

The 4-line header block is exactly HEADER_WIDTH bytes, space-padded, and
rewritten in place with pwrite so the inode never changes (tail -f safe).
Segment bodies are single O_APPEND writes followed by fsync.
"""

from __future__ import annotations

import os
from pathlib import Path

from omnilingual.models import Segment

from .markdown import (
    format_english_line,
    format_segment,
    fmt_ts,
    lang_share,
)

HEADER_WIDTH = 512
_LINES = 4
_LINE_WIDTH = HEADER_WIDTH // _LINES  # 128 bytes per line incl. newline


def _pad(line: str) -> bytes:
    raw = (line + "\n").encode("utf-8")
    if len(raw) > _LINE_WIDTH:
        raw = (line[: _LINE_WIDTH - 5] + "…" + "\n").encode("utf-8")
    return raw + b" " * (_LINE_WIDTH - len(raw))


class LiveMarkdownWriter:
    """Appends segments to a growing Markdown file, header kept current."""

    heading = "## Transcript"

    def __init__(self, path: Path, *, title: str, cost_cap: float) -> None:
        self.fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        self._fd = self.fd
        self.title = title
        self.cost_cap = cost_cap
        self._segments: list[Segment] = []
        self._cost = 0.0
        self._growing = True
        os.write(self._fd, self._header())
        static = f"{self.heading}\n\n".encode("utf-8")
        os.write(self._fd, static)
        os.fsync(self._fd)

    def _header(self) -> bytes:
        n = len(self._segments)
        shares = ", ".join(
            f"{lang} {pct}%" for lang, pct in lang_share(self._segments)
        ) or "—"
        # Duration so far = end of the last appended segment.
        dur = self._segments[-1].chunk.end_s if self._segments else 0.0
        growing = " · growing" if self._growing else ""
        return b"".join(
            [
                _pad(f"# {self.title}"),
                _pad(f"Duration (so far) {fmt_ts(dur)} · {n} segments · {shares}"),
                _pad(f"Cost ₹{self._cost:.2f} (cap ₹{self.cost_cap:.0f}){growing}"),
                _pad(""),
            ]
        )

    def append_segment(self, seg: Segment, cost_delta: float = 0.0) -> None:
        body = ("\n".join(format_segment(seg))).encode("utf-8")
        os.write(self._fd, body)  # O_APPEND single write
        self._segments.append(seg)
        self._cost += cost_delta
        os.pwrite(self._fd, self._header(), 0)
        os.fsync(self._fd)

    def finalize(self) -> None:
        self._growing = False
        os.pwrite(self._fd, self._header(), 0)
        os.fsync(self._fd)

    def close(self) -> None:
        fd, self._fd = self._fd, -1
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> LiveMarkdownWriter:
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.finalize()
        finally:
            self.close()
```

```python
class LiveEnglishWriter(LiveMarkdownWriter):
    """Mirrors the English-only file; no_speech segments are a no-op."""

    heading = "## Transcript (English)"

    def append_segment(self, seg: Segment, cost_delta: float = 0.0) -> None:
        line = format_english_line(seg)
        if line is None:
            return
        body = (line + "\n\n").encode("utf-8")
        os.write(self._fd, body)
        self._segments.append(seg)
        self._cost += cost_delta
        os.pwrite(self._fd, self._header(), 0)
        os.fsync(self._fd)
```

(Note: `LiveEnglishWriter` is defined in the same file, after `LiveMarkdownWriter`. Its header shares line is computed over English-kept segments only, which matches the batch `.en.md` content policy.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/render/test_render_live.py tests/render/test_markdown.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add omnilingual/render/live.py tests/render/test_render_live.py
git commit -m "feat: add crash-safe incremental Markdown writer for live"
```

---
### Task 8: Per-chunk live transcription step plus energy calibration

`process_chunk()` mirrors the batch per-chunk STT/MT classification exactly (same statuses, same `_price` math, same `--langs` warning), but returns the cost delta and billable flag so the run loop can enforce `--max-cost` from actual API spend. `speech=False` chunks become `no_speech` with zero cost and zero API calls (energy gate). `calibrate_energy_floor()` turns the 1 s ambient probe into a speech floor: 4× ambient, never below 0.004 (spec §4).

**Files:**
- Create: `omnilingual/pipeline/live.py` (this task: `process_chunk`, `calibrate_energy_floor`; Task 9 adds `LiveOptions`/`run_live` to the same file)
- Test: `tests/pipeline/test_process_chunk.py`

**Interfaces:**
- Consumes: `Chunk`, `Segment` from `omnilingual/models.py`; `STT_FAILED_TEXT`, `_mt_cached`, `_price`, `_stt_cached` from `omnilingual/pipeline/__init__.py` (Task 1); `Settings` from `omnilingual/config.py`; `JsonCache` from `omnilingual/cache.py`
- Produces (consumed by Task 9): `process_chunk(chunk, *, speech, stt, translator, cache, settings) -> tuple[Segment, float, bool]` (segment, INR delta, billable), `calibrate_energy_floor(ambient_rms) -> float`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from omnilingual.cache import JsonCache
from omnilingual.config import load_settings
from omnilingual.http import SarvamError, TransientError
from omnilingual.models import Chunk, STTResult
from omnilingual.pipeline.live import calibrate_energy_floor, process_chunk
from tests.conftest import make_wav, raw_pcm


class FakeSTT:
    model = "fake-stt"
    mode = "transcribe"

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def transcribe(self, wav_path):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class FakeTranslator:
    model = "fake-mt"

    def __init__(self, supported=("hi-IN", "en-IN"), error=None):
        self.supported = set(supported)
        self.error = error
        self.calls = 0

    def supports(self, lang):
        return lang in self.supported

    def to_english(self, text, src_lang):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return f"[{src_lang}->en] {text}"


def _ctx(tmp_path, **kw):
    wav = tmp_path / "c.wav"
    make_wav(wav, [("tone", 2.0)])
    chunk = Chunk(idx=0, start_s=0.0, end_s=2.0, wav_path=wav)
    settings = load_settings(api_key="k")
    cache = JsonCache(tmp_path / "cache")
    return dict(chunk=chunk, cache=cache, settings=settings, **kw)


def test_ok_hindi_bills_stt_plus_mt(tmp_path):
    stt = FakeSTT(STTResult(lang="hi-IN", prob=0.9, text="Namaste"))
    tr = FakeTranslator()
    ctx = _ctx(tmp_path, speech=True, stt=stt, translator=tr)
    seg, delta, billable = process_chunk(**ctx)
    assert (seg.status, seg.english) == ("ok", "[hi-IN->en] Namaste")
    assert billable is True and delta > 0
    assert stt.calls == 1 and tr.calls == 1


def test_english_skips_translator(tmp_path):
    stt = FakeSTT(STTResult(lang="en-IN", prob=0.99, text="Hello"))
    tr = FakeTranslator()
    ctx = _ctx(tmp_path, speech=True, stt=stt, translator=tr)
    seg, delta, billable = process_chunk(**ctx)
    assert seg.status == "ok" and seg.english is None and tr.calls == 0


def test_unsupported_language_marks_mt_unsupported(tmp_path):
    stt = FakeSTT(STTResult(lang="xx-YY", prob=0.8, text="lą"))
    tr = FakeTranslator(supported=("hi-IN",))
    ctx = _ctx(tmp_path, speech=True, stt=stt, translator=tr)
    seg, delta, billable = process_chunk(**ctx)
    assert seg.status == "mt_unsupported" and seg.english is None


def test_transient_mt_failure_marks_mt_failed(tmp_path):
    stt = FakeSTT(STTResult(lang="hi-IN", prob=0.9, text="Namaste"))
    tr = FakeTranslator(error=TransientError("overloaded", status=503))
    ctx = _ctx(tmp_path, speech=True, stt=stt, translator=tr)
    seg, delta, billable = process_chunk(**ctx)
    assert seg.status == "mt_failed"


def test_stt_error_marks_stt_failed(tmp_path):
    stt = FakeSTT(error=SarvamError("boom"))
    tr = FakeTranslator()
    ctx = _ctx(tmp_path, speech=True, stt=stt, translator=tr)
    seg, delta, billable = process_chunk(**ctx)
    assert seg.status == "stt_failed" and seg.text == "[transcription failed]"


def test_empty_text_is_no_speech(tmp_path):
    stt = FakeSTT(STTResult(lang="hi-IN", prob=0.9, text="  "))
    tr = FakeTranslator()
    ctx = _ctx(tmp_path, speech=True, stt=stt, translator=tr)
    seg, delta, billable = process_chunk(**ctx)
    assert seg.status == "no_speech" and tr.calls == 0


def test_silent_chunk_makes_no_api_calls(tmp_path):
    stt = FakeSTT(STTResult(lang="hi-IN", prob=0.9, text="Namaste"))
    tr = FakeTranslator()
    ctx = _ctx(tmp_path, speech=False, stt=stt, translator=tr)
    seg, delta, billable = process_chunk(**ctx)
    assert seg.status == "no_speech" and (delta, billable) == (0.0, False)
    assert stt.calls == 0 and tr.calls == 0


def test_calibrate_energy_floor():
    assert calibrate_energy_floor(0.001) == 0.004
    assert calibrate_energy_floor(0.005) == pytest.approx(0.02)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_process_chunk.py -q`
Expected: FAIL with "No module named 'omnilingual.pipeline.live'" (or `ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
"""Live session pipeline: per-chunk STT/MT classification plus calibration."""

from __future__ import annotations

import logging

from omnilingual.cache import JsonCache
from omnilingual.config import Settings
from omnilingual.models import Chunk, Segment
from omnilingual.pipeline import (
    STT_FAILED_TEXT,
    _mt_cached,
    _price,
    _stt_cached,
)

log = logging.getLogger("omnilingual")


def calibrate_energy_floor(ambient_rms: float) -> float:
    """Speech floor from the 1 s ambient probe: 4x ambient, floor 0.004."""
    return max(ambient_rms * 4.0, 0.004)


def process_chunk(
    chunk: Chunk,
    *,
    speech: bool,
    stt,
    translator,
     cache: JsonCache,
     settings: Settings,
) -> tuple[Segment, float, bool]:
    """Transcribe + translate one sealed chunk. Returns (segment, inr, billed).

    Mirrors the batch per-chunk classification exactly. QuotaError/AuthError
    propagate (the run loop converts them into a capture-while-halted state).
    """
    if not speech:
        seg = Segment(chunk=chunk, lang="", prob=0.0, text="",
                      english=None, status="no_speech")
        return (seg, 0.0, False)
    result = _stt_cached(chunk, stt, cache)
    if result is None:
        seg = Segment(chunk=chunk, lang="unknown", prob=0.0,
                      text=STT_FAILED_TEXT, english=None, status="stt_failed")
        return (seg, _price(chunk.duration_s, 0, settings), True)
    if settings.langs and result.lang not in settings.langs:
        log.warning(
            "chunk %d: detected %s (p=%.2f) outside configured languages %s",
            chunk.idx, result.lang, result.prob, ",".join(settings.langs),
        )
    if not result.text.strip():
        seg = Segment(chunk=chunk, lang=result.lang, prob=result.prob,
                      text="", english=None, status="no_speech")
        return (seg, _price(chunk.duration_s, 0, settings), True)
    if result.lang == "en-IN":
        seg = Segment(chunk=chunk, lang="en-IN", prob=result.prob,
                      text=result.text, english=None, status="ok")
        return (seg, _price(chunk.duration_s, 0, settings), True)
    if not translator.supports(result.lang):
        seg = Segment(chunk=chunk, lang=result.lang, prob=result.prob,
                      text=result.text, english=None, status="mt_unsupported")
        return (seg, _price(chunk.duration_s, 0, settings), True)
    english = _mt_cached(result.text, result.lang, translator, cache)
    if english is None:
        seg = Segment(chunk=chunk, lang=result.lang, prob=result.prob,
                      text=result.text, english=None, status="mt_failed")
        return (seg, _price(chunk.duration_s, 0, settings), True)
    seg = Segment(chunk=chunk, lang=result.lang, prob=result.prob,
                  text=result.text, english=english, status="ok")
    return (seg, _price(chunk.duration_s, len(result.text), settings), True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_process_chunk.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add omnilingual/pipeline/live.py tests/pipeline/test_process_chunk.py
git commit -m "feat: add live per-chunk transcription step with cost deltas"
```

---

### Task 9: Threaded live run loop with ordered appender

`run_live()` wires everything: session dir + `session.json`, ambient probe fed back into the slicer (no audio lost), capture thread draining PCM unconditionally (spec §5 — an STT retry must never stall the pipe), stderr gap thread, STT/MT worker pool (`--stt-workers`, default 2), and an ordered appender that holds out-of-order completions until lower sequence numbers land. `no_speech` chunks reach the file only when they close a >30 s gap (spec §4). Cost-cap / quota / auth halt *new API calls only* — capturing and sealing continue so `--from-chunks` (Task 2) can recover the session with zero re-payment (spec §7). Two-stage Ctrl+C: first seals and finishes gracefully, second exits immediately after reaping ffmpeg (spec §7).

**Files:**
- Modify: `omnilingual/pipeline/live.py` (append `LiveOptions`, `run_live`)
- Test: `tests/pipeline/test_run_live.py`

**Interfaces:**
- Consumes: everything from Tasks 1–8, plus `LiveCapture`, `CaptureError`, `BYTES_PER_SECOND`, `parse_silence_line`, `rms` (Task 5); `LiveSlicer`, `SealedChunk` (Task 6); `LiveMarkdownWriter`, `LiveEnglishWriter` (Task 7); `process_chunk`, `calibrate_energy_floor` (Task 8); `LIVE_SESSION_KIND`, `transcribe_chunks` (Task 2); `JsonCache`; `fmt_ts`
- Produces (consumed by Task 10): `LiveOptions` dataclass (`out: Path`, `device="Omnilingual"`, `mic_only=False`, `target_s=8.0`, `max_chunk_s=28.0`, `min_chunk_s=5.0`, `noise_db=-35.0`, `stt_workers=2`, `max_cost=50.0`, `work_root: Path | None=None`, `english_only=False`), `run_live(opts, settings, stt, translator, *, status=None, capture_factory=LiveCapture) -> int` (exit code 0/1/2)

- [ ] **Step 1: Write the failing tests**

```python
import json
import re

import respx
import httpx

from omnilingual.cache import JsonCache
from omnilingual.config import load_settings
from omnilingual.models import chunks_from_json
from omnilingual.pipeline import transcribe_chunks
from omnilingual.pipeline.live import LiveOptions, run_live
from omnilingual.audio.live_capture import BYTES_PER_SECOND, CaptureError
from omnilingual.render.markdown import render
from tests.conftest import raw_pcm

STT_URL = "https://api.sarvam.ai/speech-to-text"
MT_URL = "https://api.sarvam.ai/translate"


class FakeCapture:
    """Scripted LiveCapture: 1 s PCM blocks + threshold-gated stderr lines."""

    def __init__(self, device_name, *, mic_only=False, noise_db=-35.0,
                 blocks=(), gated=(), die_after=None):
        self._blocks = list(blocks)
        self._gated = list(gated)  # (bytes_consumed_threshold, line_bytes)
        self._die_after = die_after
        self._reads = 0
        self.consumed = 0
        self.closed = False

    def open(self):
        pass

    def read(self, n):
        self._reads += 1
        if self._die_after is not None and self._reads > self._die_after:
            raise CaptureError("fake child died")
        if not self._blocks:
            raise CaptureError("fake stream exhausted")
        self.consumed += len(self._blocks[0])
        return self._blocks.pop(0)

    def __iter__(self):
        while True:
            try:
                yield self.read(BYTES_PER_SECOND)
            except CaptureError:
                return

    @property
    def stderr(self):
        import time

        pending = list(self._gated)
        while pending or not self.closed:
            ready = [ln for need, ln in pending if self.consumed >= need]
            pending = [(need, ln) for need, ln in pending if self.consumed < need]
            for ln in ready:
                yield ln
            if not pending and self.closed:
                return
            if not pending:
                return
            time.sleep(0.001)

    def close(self):
        self.closed = True


def _pcm_3x20():
    return raw_pcm([("tone", 20.0), ("silence", 1.0)] * 2 + [("tone", 20.0)])


def _blocks(pcm):
    return [pcm[i : i + BYTES_PER_SECOND]
            for i in range(0, len(pcm), BYTES_PER_SECOND)]


def _gaps():
    return [
        (21 * BYTES_PER_SECOND, b"[silencedetect @ x] silence_start: 20.0\n"),
        (21 * BYTES_PER_SECOND, b"[silencedetect @ x] silence_end: 21.0\n"),
        (42 * BYTES_PER_SECOND, b"[silencedetect @ x] silence_start: 41.0\n"),
        (42 * BYTES_PER_SECOND, b"[silencedetect @ x] silence_end: 42.0\n"),
    ]


def _route(router, langs=("hi-IN", "en-IN", "ta-IN"), texts=("Bravo", "Hello", "Vanakkam")):
    stt = router.post(STT_URL)
    queue = [
        {"request_id": f"r{i}", "transcript": t, "language_code": lg,
         "language_probability": 0.9}
        for i, (lg, t) in enumerate(zip(langs, texts))
    ]

    def stt_cb(request):
        body = queue.pop(0) if queue else queue[-1] if queue else {
            "request_id": "rx", "transcript": "Again",
            "language_code": "hi-IN", "language_probability": 0.9}
        if not queue:
            queue.append(body)
        return httpx.Response(200, json=body)

    stt.mock(side_effect=stt_cb)
    mt = router.post(MT_URL)

    def mt_cb(request):
        import json as _json

        payload = _json.loads(request.content.decode())
        src = payload["source_language_code"]
        return httpx.Response(200, json={
            "translated_text": f"[{src}->en] {payload['input']}"})

    mt.mock(side_effect=mt_cb)
    return stt, mt


def _opts(tmp_path, **kw):
    args = dict(out=tmp_path / "meeting.md")
    args.update(kw)
    return LiveOptions(**args)


@respx.mock
def test_live_run_transcribes_in_order(router, tmp_path):
    _route(router)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    code = run_live(_opts(tmp_path), load_settings(api_key="k"),
                    SarvamSTT(load_settings(api_key="k")),
                    MayuraTranslator(load_settings(api_key="k")),
                    status=lambda m: None, capture_factory=factory)
    assert code == 0
    text = (tmp_path / "meeting.md").read_text(encoding="utf-8")
    heads = re.findall(r"\*\*\[(\d\d:\d\d:\d\d) → (\d\d:\d\d:\d\d)\] (\S+)\*\*", text)
    assert [h[2] for h in heads] == ["hi-IN", "en-IN", "ta-IN"]
    assert [h[0] for h in heads] == sorted(h[0] for h in heads)
    assert "· growing" not in text  # finalized
    bodies = text.split("## Transcript\n\n", 1)[1]
    assert bodies.index("Bravo") < bodies.index("Hello") < bodies.index("Vanakkam")


@respx.mock
def test_live_matches_batch_on_same_chunks(router, tmp_path):
    _route(router)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    run_live(_opts(tmp_path), settings, SarvamSTT(settings),
             MayuraTranslator(settings), status=lambda m: None,
             capture_factory=factory)
    session = next((tmp_path / ".omnilingual").glob("live-*"))
    chunks = chunks_from_json((session / "chunks.json").read_text(encoding="utf-8"))
    assert len(chunks) >= 2
    for c in chunks:
        assert c.end_s - c.start_s <= 28.0
    before = router.call_count
    batch = transcribe_chunks(chunks, 60.0, tmp_path, settings,
                              SarvamSTT(settings), MayuraTranslator(settings),
                              JsonCache(session / "cache"))
    live_text = (tmp_path / "meeting.md").read_text(encoding="utf-8")
    live_bodies = live_text.split("## Transcript\n\n", 1)[1]
    batch_bodies = render(batch).split("## Transcript\n\n", 1)[1]
    assert live_bodies == batch_bodies  # zero new HTTP calls needed
    assert router.call_count == before  # cache served all: zero new HTTP calls


@respx.mock
def test_second_run_reuses_cache(router, tmp_path):
    _route(router)
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    pcm = _pcm_3x20()
    run_live(_opts(tmp_path, out=tmp_path / "a.md"), settings,
             SarvamSTT(settings), MayuraTranslator(settings),
             status=lambda m: None,
             capture_factory=lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps()))
    first_calls = router.call_count
    assert first_calls > 0
    run_live(_opts(tmp_path, out=tmp_path / "b.md"), settings,
             SarvamSTT(settings), MayuraTranslator(settings),
             status=lambda m: None,
             capture_factory=lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps()))
    # NOTE: live sessions use per-session caches, so the second run re-pays.
    # Cache reuse is proven at the transcribe_chunks level (Task 2 tests).
    assert router.call_count > first_calls


@respx.mock
def test_capture_death_seals_partial_and_exits_2(router, tmp_path):
    _route(router)
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    pcm = raw_pcm([("tone", 10.0)])
    msgs = []
    code = run_live(
        _opts(tmp_path), settings, SarvamSTT(settings),
        MayuraTranslator(settings), status=msgs.append,
        capture_factory=lambda *a, **k: FakeCapture(
            *a, **k, blocks=_blocks(pcm), gated=[], die_after=4))
    assert code == 2
    assert any("died" in m for m in msgs)
    text = (tmp_path / "meeting.md").read_text(encoding="utf-8")
    assert "## Transcript" in text


@respx.mock
def test_cost_cap_halts_api_but_keeps_sealing(router, tmp_path):
    _route(router)
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    pcm = _pcm_3x20()
    msgs = []
    code = run_live(
        _opts(tmp_path, max_cost=0.0001), settings, SarvamSTT(settings),
        MayuraTranslator(settings), status=msgs.append,
        capture_factory=lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps()))
    assert code == 2
    assert any("cost cap" in m for m in msgs)
    session = next((tmp_path / ".omnilingual").glob("live-*"))
    chunks = chunks_from_json((session / "chunks.json").read_text(encoding="utf-8"))
    assert len(chunks) >= 2  # sealing continued after the halt
    text = (tmp_path / "meeting.md").read_text(encoding="utf-8")
    assert "cost cap" in text


@respx.mock
def test_quota_halt_keeps_file_valid(router, tmp_path):
    router.post(STT_URL).mock(return_value=httpx.Response(402, json={"error": "quota"}))
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    pcm = _pcm_3x20()
    msgs = []
    code = run_live(
        _opts(tmp_path), settings, SarvamSTT(settings),
        MayuraTranslator(settings), status=msgs.append,
        capture_factory=lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps()))
    assert code == 2
    assert any("quota" in m for m in msgs)
    assert "## Transcript" in (tmp_path / "meeting.md").read_text(encoding="utf-8")
```

(Notes for the implementer, all normative: `SarvamSTT` posts to `{base_url}/speech-to-text` and `MayuraTranslator` to `{base_url}/translate` with the default `base_url https://api.sarvam.ai`; the STT mock queue reuses its last entry when the live chunk count exceeds 3, so order assertions use subsequence/chronology, never exact call counts. The second-run test documents that per-session caches re-pay by design — cross-run reuse is covered by Task 2's `run_from_chunks` tests, not here.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_run_live.py -q`
Expected: FAIL with "cannot import name 'LiveOptions'" (or `ImportError`).

- [ ] **Step 3: Write minimal implementation**

Append to `omnilingual/pipeline/live.py`:

```python
"""(continued) Threaded live run loop."""

import json
import os
import queue
import select
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from omnilingual.audio.live_capture import (
    BYTES_PER_SECOND,
    CaptureError,
    LiveCapture,
    parse_silence_line,
    rms,
)
from omnilingual.audio.live_slicer import LiveSlicer
from omnilingual.cache import JsonCache
from omnilingual.http import AuthError, QuotaError
from omnilingual.models import Segment
from omnilingual.pipeline import LIVE_SESSION_KIND
from omnilingual.render.live import LiveEnglishWriter, LiveMarkdownWriter
from omnilingual.render.markdown import fmt_ts


@dataclass
class LiveOptions:
    out: Path
    device: str = "Omnilingual"
    mic_only: bool = False
    target_s: float = 8.0
    max_chunk_s: float = 28.0
    min_chunk_s: float = 5.0
    noise_db: float = -35.0
    stt_workers: int = 2
    max_cost: float = 50.0
    work_root: Path | None = None
    english_only: bool = False


_HALT_TEXT = {
    "cost": "[transcription stopped: cost cap reached]",
    "quota": "[transcription stopped: API quota exceeded]",
    "auth": "[transcription stopped: authorization failed]",
}


def run_live(opts: LiveOptions, settings, stt, translator, *,
             status: Callable[[str], None] | None = None,
             capture_factory: Callable = LiveCapture) -> int:
    """Run a live session. Returns the process exit code (0/1/2)."""
    say = status or (lambda msg: None)
    root = opts.work_root or opts.out.parent / ".omnilingual"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session = root / f"live-{stamp}"
    (session / "live-chunks").mkdir(parents=True, exist_ok=True)
    (session / "session.json").write_text(json.dumps({
        "kind": LIVE_SESSION_KIND,
        "device": opts.device,
        "mic_only": opts.mic_only,
        "langs": list(settings.langs),
        "target_s": opts.target_s,
        "max_chunk_s": opts.max_chunk_s,
        "min_chunk_s": opts.min_chunk_s,
        "started_utc": stamp,
    }, indent=2), encoding="utf-8")
    cache = JsonCache(session / "cache")

    capture = capture_factory(opts.device, mic_only=opts.mic_only,
                              noise_db=opts.noise_db)
    try:
        capture.open()
    except CaptureError as exc:
        say(f"[live] capture failed: {exc}")
        return 1

    probe = capture.read(BYTES_PER_SECOND)  # 1 s ambient probe, then reused
    slicer = LiveSlicer(session, target_s=opts.target_s,
                        max_s=opts.max_chunk_s, min_s=opts.min_chunk_s,
                        energy_floor=calibrate_energy_floor(rms(probe)))

    title = "Meeting transcript — LIVE " + datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M %Z")
    main_w = LiveMarkdownWriter(opts.out, title=title, cost_cap=opts.max_cost)
    en_w = (LiveEnglishWriter(opts.out.with_suffix(".en.md"), title=title,
                              cost_cap=opts.max_cost)
            if opts.english_only else None)
    writers = [w for w in (main_w, en_w) if w is not None]

    jobs: queue.Queue = queue.Queue()
    results: dict[int, tuple[Segment, float, bool]] = {}
    cond = threading.Condition()
    mlock = threading.Lock()
    slicer_lock = threading.Lock()
    gap_box: list[tuple[float, float]] = []
    state = {"accrued": 0.0, "halt": None, "enqueued": 0, "next": 0,
             "last_kept_end": 0.0, "sealed_end": 0.0, "sealed_n": 0,
             "appended_n": 0, "appended_end": 0.0, "bad": False, "died": False}
    stop = threading.Event()
    feeding_done = threading.Event()
    capture_done = threading.Event()
    t0 = time.monotonic()

    def elapsed() -> str:
        return fmt_ts(time.monotonic() - t0)

    def emit(sealed_list) -> None:
        with mlock:
            for sc in sealed_list:
                jobs.put((state["enqueued"], sc))
                state["enqueued"] += 1
                state["sealed_n"] += 1
                state["sealed_end"] = max(state["sealed_end"], sc.chunk.end_s)
        with cond:
            cond.notify_all()

    def halt_once(kind: str, msg: str) -> None:
        with mlock:
            if state["halt"] is None:
                state["halt"] = kind
                say(msg)

    def capture_loop() -> None:
        try:
            with slicer_lock:
                emit(slicer.feed(probe))
            for block in capture:
                if stop.is_set():
                    break
                with slicer_lock:
                    for start, end in gap_box:
                        slicer.note_gap(start, end)
                    del gap_box[:]
                    sealed = slicer.feed(block)
                emit(sealed)
        except CaptureError as exc:
            with mlock:
                state["died"] = True
            say(f"[live {elapsed()}] capture died: {exc}")
        finally:
            feeding_done.set()
            capture.close()

    def stderr_loop() -> None:
        raw = capture.stderr
        if raw is None:
            return
        pending_start: float | None = None
        if hasattr(raw, "readline"):  # real pipe: poll until feeding ends
            while True:
                ready, _, _ = select.select([raw], [], [], 0.2)
                if not ready:
                    if feeding_done.is_set():
                        return
                    continue
                line = raw.readline()
                if not line:
                    return
                text = line.decode("utf-8", "replace") if isinstance(
                    line, bytes) else line
                parsed = parse_silence_line(text)
                if parsed is None:
                    continue
                kind, ts = parsed
                if kind == "start":
                    pending_start = ts
                elif pending_start is not None:
                    with slicer_lock:
                        gap_box.append((pending_start, ts))
                    pending_start = None
        else:  # scripted iterable (tests): drain fast
            for raw_line in raw:
                text = raw_line.decode("utf-8", "replace") if isinstance(
                    raw_line, bytes) else raw_line
                parsed = parse_silence_line(text)
                if parsed is None:
                    continue
                kind, ts = parsed
                if kind == "start":
                    pending_start = ts
                elif pending_start is not None:
                    with slicer_lock:
                        gap_box.append((pending_start, ts))
                    pending_start = None

    def worker() -> None:
        while True:
            job = jobs.get()
            if job is None:
                return
            seq, sc = job
            with mlock:
                halt = state["halt"]
                over = state["accrued"] >= opts.max_cost
            if halt is not None or over:
                if halt is None:
                    halt_once("cost",
                              f"[live {elapsed()}] cost cap ₹{opts.max_cost:.0f} "
                              "reached; capturing continues, "
                              "run --from-chunks to recover")
                    halt = "cost"
                seg = Segment(chunk=sc.chunk, lang="unknown", prob=0.0,
                              text=_HALT_TEXT[halt], english=None,
                              status="stt_failed")
                item = (seg, 0.0, False)
            else:
                try:
                    item = process_chunk(sc.chunk, speech=sc.speech, stt=stt,
                                         translator=translator, cache=cache,
                                         settings=settings)
                except QuotaError:
                    halt_once("quota",
                              f"[live {elapsed()}] API quota exceeded (402); "
                              "capturing continues, run --from-chunks to recover")
                    seg = Segment(chunk=sc.chunk, lang="unknown", prob=0.0,
                                  text=_HALT_TEXT["quota"], english=None,
                                  status="stt_failed")
                    item = (seg, 0.0, False)
                except AuthError:
                    halt_once("auth",
                              f"[live {elapsed()}] authorization failed; "
                              "capturing continues")
                    seg = Segment(chunk=sc.chunk, lang="unknown", prob=0.0,
                                  text=_HALT_TEXT["auth"], english=None,
                                  status="stt_failed")
                    item = (seg, 0.0, False)
            with cond:
                results[seq] = item
                cond.notify_all()

    def status_line(seq: int, seg: Segment) -> str:
        dur = seg.chunk.duration_s
        idx = seg.chunk.idx if seg.chunk.idx >= 0 else seq
        if seg.status == "ok" and seg.lang != "en-IN":
            tail = f"→ {seg.lang} {seg.prob:.2f} → en ✓"
        elif seg.status == "ok":
            tail = f"→ {seg.lang} {seg.prob:.2f} ✓"
        elif seg.status == "no_speech":
            tail = "→ silence"
        else:
            tail = f"→ {seg.status}"
        return f"[live {elapsed()}] sealed #{idx} ({dur:.1f} s) {tail}"

    threads = [threading.Thread(target=capture_loop, daemon=True),
               threading.Thread(target=stderr_loop, daemon=True)]
    workers = [threading.Thread(target=worker, daemon=True)
               for _ in range(max(1, opts.stt_workers))]
    for th in threads + workers:
        th.start()

    # Two-stage Ctrl+C: first stops gracefully, second reaps ffmpeg and exits.
    sigints = 0
    prev = signal.getsignal(signal.SIGINT)

    def on_sigint(signum, frame):
        nonlocal sigints
        sigints += 1
        if sigints == 1:
            say(f"[live {elapsed()}] stopping… (Ctrl+C again to quit now)")
            stop.set()
        else:
            capture.close()
            os._exit(2)

    signal.signal(signal.SIGINT, on_sigint)
    last_lag_log = t0
    try:
        while True:
            with cond:
                while (state["next"] not in results
                       and not (feeding_done.is_set()
                                and state["next"] >= state["enqueued"])):
                    cond.wait(timeout=0.5)
                if state["next"] in results:
                    item = results.pop(state["next"])
                    state["next"] += 1
                elif feeding_done.is_set():
                    # Drain gaps the stderr thread parsed after feeding ended.
                    pending = None
                    with slicer_lock:
                        if gap_box:
                            pending = list(gap_box)
                            del gap_box[:]
                    if pending:
                        with slicer_lock:
                            tail = []
                            for start, end in pending:
                                tail += slicer.note_gap(start, end)
                            tail += slicer.flush()
                        emit(tail)
                        continue
                    with slicer_lock:
                        tail = slicer.flush()
                    emit(tail)
                    for _ in workers:
                        jobs.put(None)
                    for th in threads + workers:
                        th.join(timeout=30)
                    break
                else:
                    continue
            seg, delta, _ = item
            say(status_line(state["next"] - 1, seg))
            keep = True
            if (seg.status == "no_speech"
                    and seg.chunk.end_s - state["last_kept_end"] <= 30.0):
                keep = False
            if keep:
                for w in writers:
                    w.append_segment(seg, delta)
                with mlock:
                    state["accrued"] += delta
                    state["appended_n"] += 1
                    state["appended_end"] = seg.chunk.end_s
                    state["last_kept_end"] = seg.chunk.end_s
                    if seg.status != "ok":
                        state["bad"] = True
            now = time.monotonic()
            if now - last_lag_log >= 30.0:
                last_lag_log = now
                with mlock:
                    lag = state["sealed_end"] - state["appended_end"]
                    say(f"[live {elapsed()}] sealed {state['sealed_n']} · "
                        f"appended {state['appended_n']} · lag ~{lag:.0f} s")
    finally:
        signal.signal(signal.SIGINT, prev)
        for w in writers:
            try:
                w.finalize()
            finally:
                w.close()
    if state["died"]:
        say(f"[live {elapsed()}] session ended early (capture died); "
            "file remains valid")
    if state["halt"] == "cost":
        say(f"[live {elapsed()}] stopped API calls at the ₹{opts.max_cost:.0f} cap; "
            "sealed chunks remain on disk for --from-chunks recovery")
    return 2 if (state["bad"] or state["died"]) else 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_run_live.py -q`
Expected: PASS. If the order test flakes (gap thread vs. flush race), run it 5×: `for i in 1 2 3 4 5; do uv run pytest tests/pipeline/test_run_live.py::test_live_run_transcribes_in_order -q || break; done` — all 5 must pass; if any fails, fix the race (drain `gap_box` before `flush`, as written) rather than weakening the test.

- [ ] **Step 5: Commit**

```bash
git add omnilingual/pipeline/live.py tests/pipeline/test_run_live.py
git commit -m "feat: add threaded live run loop with ordered appender"
```

---
### Task 10: `live` command and `transcribe --from-chunks`

CLI wiring only — no new pipeline logic. `omnilingual live` validates flags (chunk rule first, before touching ffmpeg, mirroring `transcribe()`), resolves `--out` collisions to `<stem>-<HHMM>.md` (never truncates, spec §8), supports `--check-audio` (device list + 1 s RMS probe, zero API calls), and delegates to `run_live()` with stderr-only status lines. `transcribe --from-chunks <session-dir>` finishes an interrupted live session through `run_from_chunks()` (Task 2), reusing the session cache with zero re-payment.

**Files:**
- Modify: `omnilingual/cli.py`
- Test: `tests/test_cli_live.py`

**Interfaces:**
- Consumes: existing `app`, `console`, `_fail`, `MAX_CHUNK_LIMIT_S`, `escape` in `cli.py`; `ensure_ffmpeg`, `FfmpegMissingError` (normalize); `load_settings`, `ConfigError` (config); `SarvamSTT`, `MayuraTranslator`; `JsonCache`; `transcribe_chunks`, `run_from_chunks`, `estimate` (pipeline); `render`, `render_english_only`, `fmt_ts`; `LiveOptions`, `run_live` (Task 9); `LiveCapture`, `CaptureError`, `parse_devices`, `rms`, `dbfs`, `BYTES_PER_SECOND` (Task 5)
- Produces: `omnilingual live` command; `--from-chunks` option on `transcribe`

- [ ] **Step 1: Write the failing tests**

```python
import json
import re

import pytest
from typer.testing import CliRunner

import omnilingual.cli as cli_mod
from omnilingual.cli import app
from omnilingual.models import Chunk, chunks_to_json
from tests.conftest import make_wav

runner = CliRunner()


class FakeSTT:
    model = "fake-stt"
    mode = "transcribe"

    def __init__(self, settings):
        self.calls = 0

    def transcribe(self, wav_path):
        from omnilingual.models import STTResult

        self.calls += 1
        return STTResult(lang="hi-IN", prob=0.9, text="Namaste")


class FakeTranslator:
    model = "fake-mt"

    def __init__(self, settings):
        pass

    def supports(self, lang):
        return True

    def to_english(self, text, src_lang):
        return f"[{src_lang}->en] {text}"


@pytest.fixture
def live_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_mod, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(cli_mod, "SarvamSTT", FakeSTT)
    monkeypatch.setattr(cli_mod, "MayuraTranslator", FakeTranslator)
    return tmp_path


def _session_dir(tmp_path, n=1):
    from omnilingual.pipeline import LIVE_SESSION_KIND

    session = tmp_path / "live-XYZ"
    (session / "live-chunks").mkdir(parents=True)
    chunks = []
    for i in range(n):
        wav = session / "live-chunks" / f"{i:04d}.wav"
        make_wav(wav, [("tone", 2.0)])
        chunks.append(Chunk(idx=i, start_s=float(2 * i),
                            end_s=float(2 * i + 2), wav_path=wav.resolve()))
    (session / "chunks.json").write_text(chunks_to_json(chunks), encoding="utf-8")
    (session / "session.json").write_text(json.dumps({
        "kind": LIVE_SESSION_KIND, "device": "Omnilingual"}), encoding="utf-8")
    return session


def test_live_validates_chunks_before_ffmpeg(live_cli, monkeypatch):
    from omnilingual.audio import normalize

    def boom():
        raise AssertionError("ffmpeg must not be touched on bad flags")

    monkeypatch.setattr(normalize, "ensure_ffmpeg", boom)
    monkeypatch.setattr(cli_mod, "ensure_ffmpeg", boom)
    res = runner.invoke(app, ["live", "--out", "m.md",
                              "--max-chunk-s", "30", "--min-chunk-s", "5"])
    assert res.exit_code == 1
    assert "--max-chunk-s must be < 30" in res.output


def test_live_validates_target_and_workers(live_cli):
    res = runner.invoke(app, ["live", "--out", "m.md", "--target-s", "60"])
    assert res.exit_code == 1
    assert "--target-s" in res.output
    res = runner.invoke(app, ["live", "--out", "m.md", "--stt-workers", "0"])
    assert res.exit_code == 1
    assert "--stt-workers" in res.output


def test_live_resolves_out_collision(live_cli, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli_mod, "run_live",
                        lambda opts, *a, **k: seen.setdefault("out", opts.out) or 0)
    (live_cli / "m.md").write_text("precious", encoding="utf-8")
    res = runner.invoke(app, ["live", "--out", str(live_cli / "m.md")])
    assert res.exit_code == 0
    assert re.fullmatch(r"m-\d{4}\.md", seen["out"].name)
    assert (live_cli / "m.md").read_text(encoding="utf-8") == "precious"


def test_live_passes_options_through(live_cli, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli_mod, "run_live",
                        lambda opts, *a, **k: seen.setdefault("opts", opts) or 0)
    res = runner.invoke(app, ["live", "--out", str(live_cli / "m.md"),
                              "--input", "Mic", "--mic-only",
                              "--langs", "hi-IN,ta-IN", "--target-s", "10",
                              "--max-cost", "25", "--stt-workers", "3"])
    assert res.exit_code == 0
    o = seen["opts"]
    assert (o.device, o.mic_only, o.target_s, o.max_cost,
            o.stt_workers) == ("Mic", True, 10.0, 25.0, 3)
    # --langs rides settings (like the batch path), not LiveOptions.


def test_live_check_audio_probes_without_api(live_cli, monkeypatch):
    from omnilingual.audio.live_capture import BYTES_PER_SECOND

    def fail_settings(*a, **k):
        raise AssertionError("no API settings in --check-audio")

    monkeypatch.setattr(cli_mod, "load_settings", fail_settings)

    class FakeCap:
        def __init__(self, *a, **k):
            pass

        def open(self):
            pass

        def read(self, n):
            assert n == BYTES_PER_SECOND
            return b"\x00" * n

        def close(self):
            pass

    monkeypatch.setattr(cli_mod, "LiveCapture", FakeCap)
    from omnilingual.audio.live_capture import AudioDevice

    monkeypatch.setattr(cli_mod, "parse_devices",
                        lambda text: [AudioDevice(index=2, name="Omnilingual")])
    import subprocess
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: type("R", (), {"stderr": "", "stdout": ""})())
    res = runner.invoke(app, ["live", "--out", "m.md", "--check-audio"])
    assert res.exit_code == 0
    assert "dBFS" in res.output


def test_transcribe_from_chunks_recovers_session(live_cli):
    session = _session_dir(live_cli, n=2)
    out = live_cli / "recovered.md"
    res = runner.invoke(app, ["transcribe", "--from-chunks", str(session),
                              "--out", str(out)])
    assert res.exit_code == 0, res.output
    text = out.read_text(encoding="utf-8")
    assert text.count("Namaste") == 2
    assert "> [hi-IN->en] Namaste" in text


def test_transcribe_from_chunks_rejects_bad_dir(live_cli):
    res = runner.invoke(app, ["transcribe", "--from-chunks", str(live_cli)])
    assert res.exit_code == 1
    assert "live-session" in res.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli_live.py -q`
Expected: FAIL with "No such command 'live'" / "No such option '--from-chunks'" (or `SystemExit` nonzero).

- [ ] **Step 3: Write minimal implementation**

In `omnilingual/cli.py`:

1. Add imports: `import json`, `subprocess`, `from datetime import datetime`, `from typing import Optional`, `LiveCapture, CaptureError, parse_devices, rms, dbfs, BYTES_PER_SECOND` from `omnilingual.audio.live_capture`, `chunks_from_json` alongside `Segment` from `omnilingual.models`, `run_from_chunks, LIVE_SESSION_KIND` alongside the existing names from `omnilingual.pipeline`, `LiveOptions, run_live` from `omnilingual.pipeline.live`. (Import `run_live` at module top; tests monkeypatch `cli_mod.run_live`.)
2. Change `recording` to `Annotated[Optional[Path], typer.Argument(exists=True, dir_okay=False, readable=True, help="Audio/video recording file")] = None` — typer still applies the identical exists/readable errors whenever a recording IS passed, and treats a missing one as `None` instead of failing. Then at the top of `transcribe()` (before the `out = ...` line): `if from_chunks is not None and recording is not None: _fail("--from-chunks cannot be combined with RECORDING")`; `if from_chunks is not None: _run_from_chunks(from_chunks, out=out, english_only=english_only, estimate_only=estimate_only, api_key=api_key, work_dir=work_dir)` (next sub-step; never returns — raises `typer.Exit`); `if recording is None: _fail("Missing argument 'RECORDING'.")`. All existing `test_cli.py` tests keep passing (they always pass a recording; the missing-input test asserts nonzero, and `_fail` exits 1).
3. Add option `--from-chunks: Annotated[Optional[Path], typer.Option("--from-chunks", help="Finish a halted live session dir")] = None`, plus this helper (same file, above `transcribe()`). It mirrors `transcribe()`'s own blocks — the estimate print (cli.py lines 83-84), the progress lambda (lines 92-96), the output writes (lines 109-114) and the exit-code tail (lines 116-120):

```python
def _run_from_chunks(session_dir: Path, *, out: Path | None,
                     english_only: bool, estimate_only: bool,
                     api_key: str | None, work_dir: Path | None) -> None:
    """Finish a halted live session. Never returns (raises typer.Exit)."""
    if work_dir is not None:
        _fail("--work-dir has no effect with --from-chunks (the session cache is reused)")
    try:
        if json.loads((session_dir / "session.json").read_text(
                encoding="utf-8")).get("kind") != LIVE_SESSION_KIND:
            raise ValueError(f"kind is not {LIVE_SESSION_KIND!r}")
        chunks = chunks_from_json((session_dir / "chunks.json").read_text(
            encoding="utf-8"))
        if not chunks:
            raise ValueError("live session has no sealed chunks yet")
    except (OSError, ValueError) as exc:
        _fail(f"{session_dir} is not a live-session dir ({exc})")
    assert chunks  # narrowed by the guard above
    try:
        settings = load_settings(api_key=api_key)
        settings.require_key()
    except ConfigError as exc:
        _fail(str(exc))
    if estimate_only:
        duration = sum(c.duration_s for c in chunks)
        cost = estimate(duration, chunks, settings)
        console.print(f"{escape(session_dir.name)}: {fmt_ts(duration)} audio, {len(chunks)} chunks")
        console.print(f"Projected: STT ₹{duration / 3600 * settings.stt_inr_per_hour:.2f} + MT ~₹{cost.mt_chars / 10_000 * settings.mt_inr_per_10k_chars:.2f} = ~₹{cost.inr_estimate:.2f}")
        raise typer.Exit(0)
    out_path = out or (session_dir / "transcript.md")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _fail(f"cannot create output directory {out_path.parent}: {exc}")

    def progress(i: int, n: int, seg: Segment) -> None:
        mark = "" if seg.status == "ok" else f" [yellow]{seg.status}[/yellow]"
        # seg.lang is whatever the API reported, so it is escaped like any other
        # external value; the counter and the mark are ours.
        console.print(f"\\[{i}/{n}] {fmt_ts(seg.chunk.start_s)} {escape(seg.lang)} {seg.prob:.2f}{mark}")

    transcript = run_from_chunks(session_dir, settings, SarvamSTT(settings),
                                 MayuraTranslator(settings),
                                 JsonCache(session_dir / "cache"), progress)
    out_path.write_text(render(transcript), encoding="utf-8")
    console.print(f"Wrote {escape(str(out_path))}")
    if english_only:
        en_path = out_path.with_suffix(".en.md")
        en_path.write_text(render_english_only(transcript), encoding="utf-8")
        console.print(f"Wrote {escape(str(en_path))}")

    bad = sum(1 for s in transcript.segments if s.status != "ok")
    console.print(f"{len(transcript.segments)} segments · estimated cost ₹{transcript.cost.inr_estimate:.2f}")
    if bad:
        console.print(f"[yellow]{bad} segment(s) need attention (see notes in transcript).[/yellow]")
        raise typer.Exit(2)
    raise typer.Exit(0)
```
4. Add the `live` command:

```python
@app.command()
def live(
    out: Annotated[Path, typer.Option("--out")],
    input: Annotated[str, typer.Option("--input")] = "Omnilingual",
    mic_only: Annotated[bool, typer.Option("--mic-only")] = False,
    langs: Annotated[str, typer.Option("--langs")] = "",
    target_s: Annotated[float, typer.Option("--target-s")] = 8.0,
    max_chunk_s: Annotated[float, typer.Option("--max-chunk-s")] = 28.0,
    min_chunk_s: Annotated[float, typer.Option("--min-chunk-s")] = 5.0,
    noise_db: Annotated[float, typer.Option("--noise-db")] = -35.0,
    stt_workers: Annotated[int, typer.Option("--stt-workers")] = 2,
    max_cost: Annotated[float, typer.Option("--max-cost")] = 50.0,
    check_audio: Annotated[bool, typer.Option("--check-audio")] = False,
    english_only: Annotated[bool, typer.Option("--english-only")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
    work_dir: Annotated[Path | None, typer.Option("--work-dir")] = None,
    verbose: Annotated[bool, typer.Option("-v")] = False,
):
    """Transcribe mic + system audio live into a growing Markdown file."""
    err = Console(stderr=True)
    say = lambda m: err.print(m, markup=False)  # brackets must not render
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)
    if not 0 < min_chunk_s < max_chunk_s < MAX_CHUNK_LIMIT_S or max_chunk_s < 2 * min_chunk_s:
        _fail("--max-chunk-s must be < 30 and > --min-chunk-s, and at least 2x --min-chunk-s")
    if not min_chunk_s <= target_s <= max_chunk_s:
        _fail("--target-s must be between --min-chunk-s and --max-chunk-s")
    if stt_workers < 1:
        _fail("--stt-workers must be >= 1")
    try:
        ensure_ffmpeg()
    except FfmpegMissingError as exc:
        _fail(str(exc))
    if check_audio:
        listing = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            capture_output=True, text=True)
        devices = parse_devices((listing.stderr or "") + (listing.stdout or ""))
        if not devices:
            _fail("No AVFoundation audio devices found.")
        for d in devices:
            say(f"[{d.index}] {d.name}")
        cap = LiveCapture(input, mic_only=mic_only, noise_db=noise_db)
        try:
            cap.open()
            with cap:
                level = dbfs(rms(cap.read(BYTES_PER_SECOND)))
        except CaptureError as exc:
            _fail(str(exc))
        say(f"input '{input}': 1 s probe {level:.1f} dBFS")
        if level < -50:
            say("near silence: set the system output to the Multi-Output "
                "Device (headphones + BlackHole) and confirm the Aggregate "
                "Device 'Omnilingual' contains mic + BlackHole")
        return
    out_path = out
    if out_path.exists():
        stamp = datetime.now().strftime("%H%M")
        cand = out_path.with_name(f"{out_path.stem}-{stamp}{out_path.suffix}")
        n = 2
        while cand.exists():
            cand = out_path.with_name(
                f"{out_path.stem}-{stamp}-{n}{out_path.suffix}")
            n += 1
        say(f"--out exists; writing {cand.name} instead")
        out_path = cand
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lang_list = [s.strip() for s in langs.split(",") if s.strip()] if langs else []
    try:
        settings = load_settings(api_key=api_key, langs=lang_list)
        settings.require_key()
    except ConfigError as exc:
        _fail(str(exc))
    opts = LiveOptions(
        out=out_path, device=input, mic_only=mic_only,
        target_s=target_s, max_chunk_s=max_chunk_s, min_chunk_s=min_chunk_s,
        noise_db=noise_db, stt_workers=stt_workers, max_cost=max_cost,
        work_root=work_dir, english_only=english_only)
    code = run_live(opts, settings, SarvamSTT(settings),
                    MayuraTranslator(settings), status=say)
    raise typer.Exit(code=code)
```

(`logging`, `FfmpegMissingError`, `ConfigError` are already imported in `cli.py` for `transcribe()` — reuse them. `--langs` is a comma string here because the existing `transcribe()` declares the same shape; it rides `load_settings(..., langs=...)` exactly like the batch path, so no `langs` field exists on `LiveOptions`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_live.py tests/test_cli.py -q`
Expected: PASS — old CLI tests prove `recording` still behaves (positional, exists/readable errors, estimate, quota hint).

- [ ] **Step 5: Commit**

```bash
git add omnilingual/cli.py tests/test_cli_live.py
git commit -m "feat: add live command and transcribe --from-chunks recovery"
```

---

### Task 11: Quota-recovery drill (live halt → `--from-chunks` finishes free)

Proves the spec §7 recovery story end to end: a live session halted by a 402 finishes later through `run_from_chunks()` with zero re-payment for already-transcribed chunks (shared session cache).

**Files:**
- Test: `tests/pipeline/test_recovery.py`

**Interfaces:**
- Consumes: `run_live`, `LiveOptions` (Task 9); `run_from_chunks` (Task 2); `JsonCache`; `chunks_from_json`

- [ ] **Step 1: Write the failing test**

```python
import httpx
import respx

from omnilingual.cache import JsonCache
from omnilingual.config import load_settings
from omnilingual.models import chunks_from_json
from omnilingual.pipeline import run_from_chunks
from omnilingual.pipeline.live import LiveOptions, run_live
from omnilingual.stt.sarvam import SarvamSTT
from omnilingual.translate.mayura import MayuraTranslator
from tests.conftest import raw_pcm
from tests.pipeline.test_run_live import FakeCapture, _blocks, _gaps, _pcm_3x20

STT_URL = "https://api.sarvam.ai/speech-to-text"
MT_URL = "https://api.sarvam.ai/translate"


@respx.mock
def test_quota_halt_then_from_chunks_finishes_free(router, tmp_path):
    quota = {"on": False}

    def stt_cb(request):
        if quota["on"]:
            return httpx.Response(402, json={"error": "quota"})
        quota["on"] = True  # only the first STT call succeeds; the rest halt
        return httpx.Response(200, json={
            "request_id": "r", "transcript": "Bravo",
            "language_code": "hi-IN", "language_probability": 0.9})

    def mt_cb(request):
        return httpx.Response(200, json={"translated_text": "[hi-IN->en] Bravo"})

    router.post(STT_URL).mock(side_effect=stt_cb)
    router.post(MT_URL).mock(side_effect=mt_cb)
    settings = load_settings(api_key="k")

    pcm = _pcm_3x20()
    out = tmp_path / "meeting.md"
    code = run_live(
        LiveOptions(out=out), settings, SarvamSTT(settings),
        MayuraTranslator(settings), status=lambda m: None,
        capture_factory=lambda *a, **k: FakeCapture(
            *a, **k, blocks=_blocks(pcm), gated=_gaps()))
    assert code == 2  # chunk 0 ok and cached; chunks 1+ hit the flipped quota
    calls_after_halt = router.call_count
    session = next((tmp_path / ".omnilingual").glob("live-*"))
    chunks = chunks_from_json((session / "chunks.json").read_text(encoding="utf-8"))
    assert len(chunks) >= 2
    quota["on"] = False  # quota restored: finish the halted session
    done = run_from_chunks(session, settings, SarvamSTT(settings),
                           MayuraTranslator(settings),
                           JsonCache(session / "cache"))
    assert all(s.status == "ok" for s in done.segments)
    stt_calls = [c for c in router.calls
                 if c.request.url.path == "/speech-to-text"]
    # chunk(s) transcribed before the halt were NOT re-paid:
    assert len(stt_calls) == calls_after_halt_stt + (len(chunks) - 1)
```

(Single live run by design: the in-callback flip is deterministic under the worker pool — exactly one STT call succeeds no matter the thread interleaving. The recovery assertion above is normative: chunk 0 was cached before the halt, so `run_from_chunks` pays for exactly `len(chunks) - 1` chunks.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipeline/test_recovery.py -q`
Expected: FAIL — `run_from_chunks` (Task 2) does not exist yet? It does (Task 2 done). So instead expect FAIL on the test helpers import if Task 9's `FakeCapture` is not importable (it is test-local…). Honest expectation: FAIL with `ImportError: cannot import name 'FakeCapture'` unless Task 9's test file exposes it at module level — it does (`class FakeCapture` at module top of `test_run_live.py`). If the import already works, this test FAILs on assertion until the quota path is wired (Task 9 done, so it should PASS — in that case, still run: a passing recovery drill on the first try is fine, commit it).

- [ ] **Step 3: Run test to verify it passes**

Run: `uv run pytest tests/pipeline/test_recovery.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/pipeline/test_recovery.py
git commit -m "test: prove quota-halt recovery pays zero for cached chunks"
```

---

### Task 12: Manual hardware checklist (spec §10)

The five manual marks need a real mic, BlackHole, and a paid key, so they live behind `OMNILINGUAL_MANUAL=1` in their own file — never in `tests/test_live.py` (reserved for the existing real-API tests). The module docstring is the checklist; the one automated test is the `--check-audio` smoke test that runs only under the manual gate.

**Files:**
- Test: `tests/test_live_manual.py`

**Interfaces:**
- Consumes: `app` from `omnilingual/cli.py`; `pytest.mark.live` marker (already registered in `pyproject.toml`)

- [ ] **Step 1: Write the test file**

```python
"""Manual live-session checklist (spec §10). NOT run by default.

Run on a Mac with mic + BlackHole + SARVAM_API_KEY set::

    OMNILINGUAL_MANUAL=1 uv run pytest tests/test_live_manual.py -q

Marks:
1. Mic loopback (30 s): `uv run omnilingual live --out loop.md`, speak Hindi
   ~30 s, Ctrl+C once. Expect: 2-4 segments, Hindi text + English quotes,
   exit 0, `tail -f loop.md` grew live.
2. One real STT+MT round trip is covered by mark 1 (no mocks involved).
3. TCC prompt path: `tccutil reset Microphone`, re-run mark 1, expect the
   Settings ▸ Privacy ▸ Microphone hint and exit 1 (no traceback).
4. BlackHole clip diff: play a fixed clip through BlackHole twice — once via
   `live --mic-only` off (Aggregate) into live.md, once transcribed from a
   file recording. Expect: same sentences, timestamps within ±2 s.
5. 48 kHz mismatch: set BlackHole to 48 kHz in Audio MIDI Setup, run
   `--check-audio`. Expect: probe still reports a dBFS level (aresample
   absorbs it) or a clear device error — never silent garbage.
"""

import os

import pytest
from typer.testing import CliRunner

from omnilingual.cli import app

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("OMNILINGUAL_MANUAL") != "1",
        reason="manual only: needs mic, BlackHole, and SARVAM_API_KEY"),
]

runner = CliRunner()


def test_check_audio_smoke():
    res = runner.invoke(["live", "--out", "manual.md", "--check-audio"])
    assert res.exit_code == 0, res.output
    assert "dBFS" in res.output
```

- [ ] **Step 2: Verify the gate works both ways**

Run: `uv run pytest tests/test_live_manual.py -q`
Expected: PASS via 1 skip (gate closed, suite stays green).

Run: `OMNILINGUAL_MANUAL=1 uv run pytest tests/test_live_manual.py -q`
Expected on a Mac WITHOUT devices: FAIL on the smoke test with a device error (proves the test really probes hardware); on a provisioned Mac: PASS. Either way the gate, not the mock, decides.

- [ ] **Step 3: Commit**

```bash
git add tests/test_live_manual.py
git commit -m "test: add manual hardware checklist for live sessions"
```



