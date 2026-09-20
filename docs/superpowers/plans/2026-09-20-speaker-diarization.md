# Speaker Diarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Label every transcript segment with "Speaker 1", "Speaker 2", … in both `transcribe` and `live` output, using a pluggable diarizer (sherpa-onnx default, pyannote optional).

**Architecture:** A new `omnilingual/diarize/` package mirrors the STT provider pattern: a `Diarizer` protocol, a sherpa-onnx implementation, a `build_diarizer` factory, and an `OnlineSpeakerTracker` for live mode. Batch diarizes the normalized recording once and maps turns onto chunks; live clusters per-chunk embeddings online. Rendering changes only in `format_segment`, which batch and live already share.

**Tech Stack:** sherpa-onnx (Apache-2.0, ONNX segmentation + 3dspeaker ERes2Net embeddings), soundfile, httpx (model download), pytest, respx-free unit tests via injectable engines.

**Spec:** `docs/superpowers/specs/2026-09-20-speaker-diarization-design.md` (committed 707f6a0 on this branch).

---

## Global Constraints

- **Additive only.** Default behavior is byte-identical to today: diarization is opt-in via `--diarize`. Golden rendering tests must pass unmodified.
- **Dependency policy.** Core deps stay `httpx`, `typer`, `rich`. sherpa-onnx + soundfile live under a `diarize` optional extra; pyannote under a separate `diarize-pyannote` extra (Task 7).
- **Thread-safety.** Live worker threads share the diarizer/tracker; every public entry point serializes on a `threading.Lock` (same pattern as `MlxWhisperSTT`).
- **Cache discipline.** Diarization results cache in the work dir keyed by recording digest + diarizer model + num_speakers — re-runs with the same settings never re-diarize.
- **TDD.** Every task: write the failing test first, verify it fails, implement, verify it passes.
- **Commits.** Conventional style (`feat:`/`test:`/`build:`/`docs:`), implementation + its tests in the same commit.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `omnilingual/models.py` | Modify | Add `speaker: str | None = None` to `Segment` |
| `omnilingual/render/markdown.py` | Modify | Speaker label in `format_segment`; speaker share line in header |
| `omnilingual/diarize/__init__.py` | Create | `build_diarizer(settings)` factory |
| `omnilingual/diarize/base.py` | Create | `Turn` dataclass, `Diarizer` protocol |
| `omnilingual/diarize/sherpa.py` | Create | `SherpaDiarizer`, model download, `_SherpaEngine` (injectable) |
| `omnilingual/diarize/assign.py` | Create | `assign_speakers(chunks, turns)` max-overlap + first-appearance renumbering |
| `omnilingual/diarize/online.py` | Create | `OnlineSpeakerTracker` for live mode |
| `omnilingual/pipeline/__init__.py` | Modify | `_diarize_cached` + speaker assignment in `transcribe_chunks` |
| `omnilingual/pipeline/live.py` | Modify | `process_chunk` speaker via tracker |
| `omnilingual/config.py` | Modify | `Settings.diarizer`, `Settings.num_speakers`, validation |
| `omnilingual/cli.py` | Modify | `--diarize`, `--speakers N`, `--diarizer` flags on `transcribe` and `live` |
| `pyproject.toml` | Modify | `diarize` extra, `diarize` pytest marker |
| `tests/conftest.py` | Modify | `--run-diarize` opt-in flag, auto-skip `diarize` marks |
| `tests/diarize/*` | Create | Unit + manual tests |
| `scripts/eval_diarize.py` | Create | Manual eval harness |
| `README.md` | Modify | Diarization usage + provider table row |

---

### Task 1: Segment.speaker + rendering (foundation, no diarizer yet)

**Files:**
- Modify: `omnilingual/models.py` (Segment)
- Modify: `omnilingual/render/markdown.py`
- Test: `tests/test_render.py` (or existing render tests), `tests/test_pipeline.py`

- [ ] **Step 1: failing tests** — segment with `speaker="Speaker 2"` renders head as `**[0:12 → 0:40] Speaker 2 · hi-IN**`; `speaker=None` renders exactly as today; english-only render prefixes `Speaker 2: `; header gains `Speakers: Speaker 1 42% · Speaker 2 31% · Speaker 3 27%` line only when any segment has a speaker.
- [ ] **Step 2: verify fail** — `uv run pytest tests/test_render.py -q` shows the new assertions failing (Segment has no `speaker`).
- [ ] **Step 3: implement** — `Segment` gains `speaker: str | None = None` (keyword, default keeps every existing construction site valid); `format_segment` inserts `Speaker N · ` before lang when set; english-only path prefixes `Speaker N: `; header computes share from summed chunk durations per speaker, mirroring `lang_share` (render/markdown.py:24).
- [ ] **Step 4: verify pass** — new tests pass AND the existing golden rendering tests pass unmodified (proves additive).
- [ ] **Step 5: commit** — `feat: add optional speaker labels to transcript rendering`

### Task 2: diarize extra + opt-in test marker

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`

- [ ] **Step 1: failing test** — `tests/test_config.py` (or a packaging test): `importlib.util.find_spec("sherpa_onnx")` guard helper `diarize_available()` returns False in base env (document, don't assert environment specifics — instead unit-test the guard logic with a monkeypatched find_spec, matching the mlx/faster-whisper pattern in tests/stt/).
- [ ] **Step 2: implement pyproject** — `[project.optional-dependencies] diarize = ["sherpa-onnx>=1.12,<2", "soundfile>=0.12"]`; pytest marker `diarize: runs real diarization models; needs the diarize extra and --run-diarize`.
- [ ] **Step 3: implement conftest** — add `--run-diarize` flag and extend `pytest_collection_modifyitems` to auto-skip `diarize`-marked tests unless the flag is present (copy the existing `--run-faster-whisper` pattern exactly).
- [ ] **Step 4: verify** — `uv run pytest -q` → full suite green with 5 skipped unchanged (no diarize tests exist yet, so skip count must not move).
- [ ] **Step 5: commit** — `build: add diarize extra and opt-in test marker`

### Task 3: omnilingual/diarize package — base + sherpa implementation

**Files:**
- Create: `omnilingual/diarize/__init__.py`, `base.py`, `sherpa.py`
- Test: `tests/diarize/__init__.py`, `tests/diarize/test_sherpa.py`, `tests/diarize/test_factory.py`

- [ ] **Step 1: failing tests** — (a) `build_diarizer` with `diarizer="sherpa"` and missing extra raises `ConfigError` containing `uv sync --extra diarize` (monkeypatch `find_spec` → None, same pattern as `test_mlx_whisper.py` missing-extra test); (b) `SherpaDiarizer.model` is namespaced `sherpa:<segmentation>-<embedding>`; (c) `transcribe`-equivalent `diarize(wav)` maps engine output to `Turn` list and is serialized through the lock (fake engine, two threads, assert no interleaving — mirror the mlx lock test); (d) `ensure_models` triggers download only when files absent (fake downloader recording calls).
- [ ] **Step 2: verify fail** — `uv run pytest tests/diarize -q` fails on import.
- [ ] **Step 3: implement `base.py`** —

```python
@dataclass(frozen=True)
class Turn:
    start_s: float
    end_s: float
    speaker: str  # raw engine label, e.g. "0", "1"

class Diarizer(Protocol):
    model: str
    def diarize(self, wav_path: Path) -> list[Turn]: ...
```

- [ ] **Step 4: implement `sherpa.py`** — `SherpaDiarizer(settings, engine=None, downloader=None)`: `model = f"sherpa:{SEGMENTATION}-{EMBEDDING}"`; constants `SEGMENTATION = "pyannote-segmentation-3-0"`, `EMBEDDING = "3dspeaker-eres2net"`; `ensure_models()` downloads from k2-fsa GitHub release tarballs into `~/.cache/omnilingual/models/` via `httpx.stream` + `tarfile` (atomic: extract to tmp dir, rename); lazy `_SherpaEngine` wraps `sherpa_onnx.OfflineSpeakerDiarization` (segmentation + embedding + FastClustering config, `num_clusters=settings.num_speakers or None`); `diarize()` acquires `threading.Lock`, calls engine, converts to `Turn`s; `find_spec("sherpa_onnx")` guard with the standard `uv sync --extra diarize` hint.
- [ ] **Step 5: implement `__init__.py`** — `build_diarizer(settings) -> Diarizer`: `"sherpa"` → `SherpaDiarizer`, unknown → `ConfigError` listing known diarizers (mirror `build_stt`).
- [ ] **Step 6: verify pass** — `uv run pytest tests/diarize -q` green (all unit-level, no model downloads).
- [ ] **Step 7: real-model smoke test** — `tests/diarize/test_sherpa.py::test_real_engine_smoke` marked `diarize`: gated `pytest.importorskip("sherpa_onnx")`, synthesizes a 2-speaker wav (two tones/voice-like noise bursts alternating 5 s) into tmp_path, asserts ≥1 turn and ≤4 distinct labels. Runs only under `--run-diarize`.
- [ ] **Step 8: commit** — `feat: add sherpa-onnx diarizer with model download and factory`

### Task 4: batch pipeline — diarize once, assign per chunk

**Files:**
- Create: `omnilingual/diarize/assign.py`
- Modify: `omnilingual/pipeline/__init__.py`, `omnilingual/cli.py`, `omnilingual/config.py`
- Test: `tests/diarize/test_assign.py`, `tests/test_pipeline.py`, `tests/test_cli.py`

- [ ] **Step 1: failing tests (assign.py, pure logic)** — table-driven cases: (a) a turn covering 80% of a chunk wins; (b) tie (exactly 50/50 overlap) resolves to the previous chunk's speaker; (c) chunk overlapping no turn → `speaker=None`; (d) first-appearance renumbering maps raw labels {"4","2","4"} → {"Speaker 1","Speaker 2","Speaker 1"} in chronological chunk order; (e) overlapping turns choose the larger overlap; (f) empty turn list → all None.
- [ ] **Step 2: failing tests (pipeline)** — fake `Diarizer` returning fixed turns; `transcribe_chunks` with diarizer labels each segment; cache path: second run with a *different* fake diarizer (same model string) still returns cached turns (proves diarization cached); cache key differs when `num_speakers` changes.
- [ ] **Step 3: verify fail** — `uv run pytest tests/diarize/test_assign.py tests/test_pipeline.py -q`.
- [ ] **Step 4: implement `assign.py`** — `assign_speakers(chunks: list[Chunk], turns: list[Turn]) -> dict[int, str | None]`: for each chunk, overlap-seconds per turn = `max(0, min(end)-max(start))`; pick max; ties → previous chunk's assigned speaker; zero overlap → None. Then `renumber(labels)` → first-appearance `Speaker N`.
- [ ] **Step 5: implement pipeline hook** — `transcribe_chunks(..., diarizer: Diarizer | None = None)`: when set, `_diarize_cached(source_wav, diarizer, work_dir, settings)` reads/writes `diarize-<model-slug>.json` in the work dir (payload: model, num_speakers, turns); turns feed `assign_speakers`; each `Segment` gets `speaker=` at construction. Wire `run()` and `run_from_chunks()` to accept and pass the diarizer (run_from_chunks re-diarizes the session's concatenated sealed chunks: live recovery gets *authoritative offline* labels — spec §5).
- [ ] **Step 6: config + CLI** — `Settings`: `diarizer: str | None = None`, `num_speakers: int | None = None`; validation: `num_speakers` ≥ 2 when set; unknown diarizer name → ConfigError. CLI `transcribe`: `--diarize` (sets `diarizer="sherpa"`), `--speakers N`, `--diarizer {sherpa}`; progress line gains `· Speaker 2` when present; `--estimate` unaffected.
- [ ] **Step 7: verify pass** — full suite green; existing pipeline/cli tests untouched and passing (diarizer defaults to None everywhere).
- [ ] **Step 8: commit** — `feat: assign speakers to batch transcript segments via diarization`

### Task 5: live mode — online speaker tracking

**Files:**
- Create: `omnilingual/diarize/online.py`
- Modify: `omnilingual/pipeline/live.py`, `omnilingual/cli.py`, `omnilingual/diarize/sherpa.py`
- Test: `tests/diarize/test_online.py`, `tests/test_live.py`

- [ ] **Step 1: failing tests (tracker, fake extractor)** — (a) first chunk → "Speaker 1"; (b) near-identical embedding (cosine > 0.6) → same speaker; (c) orthogonal embedding → "Speaker 2"; (d) after many chunks, label count stays bounded (≤ max_speakers, default 32); (e) centroid update moves toward the running mean (assert with simple vectors); (f) thread-safety: 2 threads × 50 assignments → no duplicate "Speaker N" creation races (unique labels = unique creations).
- [ ] **Step 2: failing test (live wiring)** — `process_chunk` with a stub tracker sets `seg.speaker`; speech=False chunks keep `speaker=None` (no embedding work on silence).
- [ ] **Step 3: verify fail** — `uv run pytest tests/diarize/test_online.py tests/test_live.py -q`.
- [ ] **Step 4: implement `online.py`** —

```python
class OnlineSpeakerTracker:
    def __init__(self, extractor, threshold: float = 0.6,
                 floor: float = 0.45, update: float = 0.2,
                 max_speakers: int = 32) -> None: ...
    def assign(self, wav_path: Path) -> str: ...  # "Speaker N", lock-held
```

  `extractor` is a `sherpa_onnx.SpeakerEmbeddingExtractor` wrapper (injectable for tests); cosine similarity against per-speaker centroids; score ≥ threshold → match; best score in [floor, threshold) → match only if it also clears `threshold - margin` (hysteresis against flip-flopping); else new centroid. Centroid update: `c ← (1-update)·c + update·e`, renormalized.
- [ ] **Step 5: wire live** — `run_live(..., diarizer=None)`: construct one `OnlineSpeakerTracker` from the sherpa engine when `--diarize`; `process_chunk(..., tracker=None)` calls `tracker.assign(chunk.wav_path)` after speech=True classification; `LiveOptions` unchanged shape (diarizer passed like `stt`/`translator`). CLI `live`: same three flags; help text notes `--stt-workers 1` recommended with `--diarize` (embeddings serialize on the lock anyway).
- [ ] **Step 6: verify pass** — suite green; `tests/test_live.py` existing tests pass unmodified (tracker defaults None).
- [ ] **Step 7: commit** — `feat: label live transcript segments with online speaker tracking`

### Task 6: manual test harness + real-audio gate

**Files:**
- Create: `tests/diarize/test_manual.py`, `scripts/eval_diarize.py`
- Modify: `tests/conftest.py` (if a `manual` marker is needed — else reuse env gate)

- [ ] **Step 1: manual tests** — `tests/diarize/test_manual.py`, every test skipped unless `OMNILINGUAL_MANUAL=1`: (a) two-speaker synthetic wav (alternating 5 s tones at 220 Hz / 440 Hz with noise) → exactly 2 speakers, no flips longer than 1 chunk; (b) silence-only wav → no crash, zero/None speakers; (c) 3-speaker synthetic → 3 speakers with `--speakers 3`.
- [ ] **Step 2: eval script** — `scripts/eval_diarize.py RECORDING`: runs `transcribe --diarize`, prints per-speaker line/word counts and the first 3 segments per speaker for eyeballing; exit non-zero if a single speaker owns 100% of a multi-speaker recording (crude collapse detector).
- [ ] **Step 3: verify** — `OMNILINGUAL_MANUAL=1 uv run pytest tests/diarize/test_manual.py -q` on this Intel Mac (sherpa runs on CPU/onnxruntime — already proven by the faster-whisper install); script runs against a real archived meeting recording.
- [ ] **Step 4: commit** — `test: add manual diarization gate and eval harness`

### Task 7 (optional, deferred): pyannote community-1 provider

**Files:**
- Create: `omnilingual/diarize/pyannote_impl.py`
- Modify: `omnilingual/diarize/__init__.py`, `pyproject.toml`

- [ ] **Step 1:** extra `diarize-pyannote = ["pyannote.audio>=4.0"]`; `--diarizer pyannote` requires `HF_TOKEN` env (gated model) and raises `ConfigError` otherwise; torch CPU on the Intel Mac runs ≈1× RT, so this stays a quality comparison backend, never the default.
- [ ] **Step 2:** tests mirror the sherpa suite with a fake pipeline object; `find_spec("pyannote.audio")` guard.
- [ ] **Step 3:** commit — `feat: add optional pyannote diarizer backend`
- [ ] **Gate:** only start this if the Task 6 gate shows sherpa quality is insufficient on real recordings (spec R-1).

### Task 8: README + spec status

**Files:**
- Modify: `README.md`, `docs/superpowers/specs/2026-09-20-speaker-diarization-design.md`

- [ ] **Step 1:** README — new "Speaker diarization" section: install (`uv sync --extra diarize`), `--diarize/--speakers N/--diarizer`, batch + live examples, model download location (`~/.cache/omnilingual/models/`, ~40 MB), live note (`--stt-workers 1` recommended), English-only rendering example, limits (overlapping speech, Indian-accent embedding quality gate).
- [ ] **Step 2:** spec status header → `Tasks 1–6 implemented (plan 2026-09-20-speaker-diarization.md)`.
- [ ] **Step 3:** commit — `docs: document speaker diarization usage and status`

---

## Manual Testing Checklist (end of Task 6, before merge)

- [ ] `uv sync --extra diarize` installs sherpa-onnx + soundfile on this Intel Mac (onnxruntime<1.24 pin from local-stt already present).
- [ ] Batch: `uv run omnilingual transcribe <meeting.m4a> --diarize --speakers 3` → transcript heads show `Speaker N · lang`; header shows speaker share line; re-run is instant (diarization cache hit).
- [ ] Batch english-only: `--english-only --diarize` → `.en.md` lines prefixed `Speaker N: `.
- [ ] Live: `uv run omnilingual live --stt faster-whisper --diarize --stt-workers 1 --out ./live.md` → labels appear as the file grows; two people speaking alternately get two labels.
- [ ] Recovery: stop a live session, `transcribe --from-chunks <session> --diarize` → authoritative labels, no crash on missing live cache.
- [ ] Default regression: every command above WITHOUT `--diarize` produces byte-identical output to main.

## Execution Order & Dependencies

```
T1 (render) ──► T4 (batch) ──► T5 (live) ──► T6 (manual gate) ──► T8 (docs)
T2 (extra)  ──┘        ▲
T3 (sherpa) ───────────┘
T7 (pyannote) — optional, after T6 gate fails only
```

T1→T6 are the merge bar; T7/T8-followups only if the gate demands it. All tasks land on `feat/speaker-diarization`; merge to main after the manual checklist passes.
