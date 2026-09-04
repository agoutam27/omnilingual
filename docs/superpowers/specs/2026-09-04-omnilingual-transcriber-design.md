# Omnilingual: Multilingual Indian-Language Meeting Transcriber — Design

Date: 2026-09-04
Status: Approved (v1 scope)

## 1. Goal

A Python CLI that takes a Zoom local recording of a meeting in which participants speak several Indian languages (plus English), and produces a Markdown transcript containing each spoken segment in its original language and script, followed by its English translation.

v1 is post-meeting, file-in / file-out, single-user, run on the author's Mac.

## 2. Context and constraints

- **Input**: Zoom "Record to this computer" output (`.m4a` audio or `.mp4` video). Any ffmpeg-readable container is accepted.
- **Meetings**: 30–90 minutes typical. Three to five known languages per user (e.g. `hi-IN`, `ta-IN`, `te-IN`, `kn-IN`, `en-IN`), with mid-sentence code-mixing.
- **Speech provider**: Sarvam AI.
  - Speech-to-text: Saaras v4 via `POST /speech-to-text`, `mode="transcribe"`, `language_code="unknown"` for auto-detect. Response includes detected `language_code` and `language_probability`. Synchronous endpoint accepts audio under 30 seconds per call.
  - Translation: `POST /translate`, model `mayura:v1` (default) to `en-IN`. Covers Hindi, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, Marathi, Punjabi, Odia, English. Input limited to 1000 characters per call; `sarvam-translate:v1` (configurable) allows 2000 characters and covers all 22 STT languages. Long chunk text is split at sentence boundaries before translation.
  - API access: direct REST via `httpx`, not the `sarvamai` SDK, so request shapes are explicit and tests mock HTTP with `respx`.
  - Pricing (Sept 2026): STT ₹30/hour of audio; Mayura ₹20 per 10,000 characters. Free credits are limited (₹100 stated on pricing page; another page says ₹1,000). Budget must be treated as scarce.
- **Not in v1**: live capture, speaker attribution, Notion export, AI summaries, GUI, Zoom app integration. Notion AI Meeting Notes has no public write API; a future Notion step would create a plain page via `pages.create` / `blocks.append`.

## 3. Approach chosen

Chunked synchronous pipeline: split audio into ≤28-second silence-aware chunks, transcribe each chunk with Saaras (auto-detect language per chunk), translate each chunk's text with Mayura, render Markdown. Every chunk result is cached on disk so re-runs and crash recovery cost zero API calls.

Alternatives rejected:

- **Batch API on whole file**: returns a single detected language per file, and two independent passes (transcribe, translate) segment differently, so original and English cannot be interleaved reliably.
- **Saaras double pass (transcribe + translate audio modes)**: doubles audio cost; translation not grounded in the transcript text. Kept as a possible future `Translator` implementation for languages Mayura lacks.

## 4. Architecture

```
recording.m4a
   │  ffmpeg
   ▼
audio/normalize.py   → 16 kHz mono 16-bit PCM WAV
   │
   ▼
audio/chunker.py     → list[Chunk]  (silence-aware, ≤28 s, ≥5 s)
   │
   ▼
stt/sarvam.py        → per chunk: STTResult(lang, prob, text)     [cached]
   │
   ▼
translate/mayura.py  → per segment: english text (skip if en-IN)   [cached]
   │
   ▼
render/markdown.py   → transcript.md
```

`pipeline.py` wires stages together. Stages communicate only through the dataclasses in §6 and know nothing of each other.

## 5. Components

| Module | Responsibility | Depends on |
|---|---|---|
| `omnilingual/cli.py` | Typer CLI. `omnilingual transcribe <file> [--langs ...] [--out ...] [--estimate] [--english-only]` | `pipeline` |
| `omnilingual/audio/normalize.py` | Convert any container to 16 kHz mono PCM WAV via `ffmpeg` subprocess. Fail fast if `ffmpeg` missing. | ffmpeg binary |
| `omnilingual/audio/chunker.py` | Detect silences (ffmpeg `silencedetect`), cut at the last silence before the 28 s mark; fall back to hard cut at 28 s if no silence found; merge fragments shorter than 5 s into neighbour. Write chunk WAVs. | ffmpeg binary |
| `omnilingual/stt/sarvam.py` | `SarvamSTT.transcribe(wav_path) -> STTResult`. Saaras v4, `mode=transcribe`, `language_code=unknown`. Retry policy from §7. | `httpx`, `http.py` |
| `omnilingual/translate/base.py` | `Translator` protocol: `to_english(text, src_lang) -> str \| None`; `supports(lang) -> bool`. | — |
| `omnilingual/translate/mayura.py` | Mayura implementation of `Translator`. | `httpx`, `http.py` |
| `omnilingual/cache.py` | JSON file cache. Key = `sha256(chunk wav bytes)` + model id + mode (STT) or `sha256(text)` + src lang + model (MT). | filesystem |
| `omnilingual/render/markdown.py` | `Transcript -> str`. Pure function. | — |
| `omnilingual/pipeline.py` | Orchestration, progress reporting, cost tally, resumability. | all above |
| `omnilingual/config.py` | API key from `SARVAM_API_KEY` env var or `--api-key`. Default languages, chunk limits, model ids. | — |

## 6. Data model

```python
@dataclass(frozen=True)
class Chunk:
    idx: int
    start_s: float
    end_s: float
    wav_path: Path

@dataclass(frozen=True)
class STTResult:
    lang: str            # e.g. "hi-IN"
    prob: float          # language_probability from API, 0..1
    text: str            # original script

@dataclass
class Segment:
    chunk: Chunk
    lang: str
    prob: float
    text: str
    english: str | None  # None when lang is en-IN, unsupported, or MT failed
    status: Literal["ok", "stt_failed", "mt_unsupported", "mt_failed"]

@dataclass
class Cost:
    audio_seconds: float
    mt_chars: int
    inr_estimate: float

@dataclass
class Transcript:
    source: Path
    duration_s: float
    segments: list[Segment]
    cost: Cost
```

### Work directory

`<out_dir>/.omnilingual/<sha256(source file)[:12]>/`

```
normalized.wav
chunks/0000.wav … NNNN.wav
chunks.json                 # serialized list[Chunk]
cache/stt/<key>.json
cache/mt/<key>.json
```

Re-running the same command on the same file performs zero API calls once all chunks are cached.

## 7. Error handling and resumability

| Situation | Behaviour |
|---|---|
| `ffmpeg` not on PATH | Exit before any API call with install hint (`brew install ffmpeg`). |
| API 429 or 5xx | Retry up to 3 times, exponential backoff (1 s, 2 s, 4 s) with jitter. |
| API 401/403 | Exit immediately: bad key. |
| API 402 or quota-exhausted response | Persist progress, exit with message: re-run same command to resume. |
| Chunk STT fails after retries | `Segment.status = "stt_failed"`, text `"[transcription failed]"`, continue. |
| Detected lang not supported by Mayura | `english = None`, `status = "mt_unsupported"`, render original with note. |
| Mayura fails after retries | `english = None`, `status = "mt_failed"`. |
| Detected lang outside `--langs` set | Accept detection, log warning with chunk index and probability. No re-run. |
| Ctrl-C | Cache already written for completed chunks; re-run resumes. |

`--estimate` runs normalize + chunk only and prints: chunk count, total audio minutes, projected STT cost, projected MT cost assuming ~15 characters per second of speech. No API calls.

## 8. Output format

`transcript.md`, interleaved layout:

```markdown
# Meeting transcript — recording.m4a

Duration 01:12:40 · 146 segments · Languages: hi-IN 61%, ta-IN 22%, en-IN 17%
Estimated cost: ₹58.20

## Transcript

**[00:00:04 → 00:00:29] hi-IN**
हम आज payment dashboard के बारे में बात करेंगे...
> We will talk about the payment dashboard today...

**[00:00:29 → 00:00:51] en-IN**
Okay, let's start with the refund numbers.

**[00:00:51 → 00:01:18] ta-IN** _(translation unavailable)_
...
```

Rules:

- English-detected segments show the original line only; no duplicate quote.
- `status != "ok"` segments carry an italic note after the header.
- `--english-only` emits a second file `transcript.en.md` containing just the English lines (original text for `en-IN`, translation otherwise), one paragraph per segment, no timestamps. Segments without English are rendered as `[hi-IN, untranslated]`.
- Language percentages are by audio duration, not segment count.

## 9. CLI

```
omnilingual transcribe RECORDING
    --langs hi-IN,ta-IN,en-IN     hint set; used for warnings only in v1
    --out transcript.md           default: <recording stem>.md next to input
    --english-only                also write <stem>.en.md
    --estimate                    chunk and price, no API calls
    --api-key KEY                 overrides SARVAM_API_KEY
    --work-dir PATH               default: <out dir>/.omnilingual
    --max-chunk-s 28  --min-chunk-s 5
```

Progress: one line per chunk (`[12/146] 00:05:36 hi-IN 0.97`), final summary with cost. Exit code 0 on success, 2 on partial success (any segment not `ok`), 1 on fatal.

Implementation note (2026-09-04): single root command; invocation is `omnilingual RECORDING`, no `transcribe` subcommand.

## 10. Testing

- **Unit**
  - `chunker`: synthetic WAV with known silence gaps → boundaries fall on gaps, no chunk > 28 s, none < 5 s, full coverage with no overlap.
  - `cache`: miss → store → hit; different model id → different key.
  - `render/markdown`: golden-file comparison for interleaved and english-only outputs, including every `status` value.
  - retry: mocked 429 then 200 → one result, two attempts; 401 → immediate raise.
- **Integration**: `SarvamSTT` and `MayuraTranslator` against `respx`-mocked HTTP, recorded from real responses. One `@pytest.mark.live` test per adapter with a bundled 10-second clip, skipped unless `SARVAM_API_KEY` set.
- **End-to-end**: 60-second synthetic three-language clip → full pipeline with mocked API → golden Markdown. Second run asserts zero HTTP calls (cache).
- Development follows TDD per module.

Implementation note (2026-09-04): test WAVs are synthesized in-process by `tests/conftest.make_wav` (tone/silence parts, configurable rate and channels) rather than checked in, so there is no `tests/fixtures/` directory. Golden Markdown lives in `tests/render/golden/`; recorded API JSON is inlined in the `respx` tests.

## 11. Project layout

```
omnilingual/
  pyproject.toml         # uv / hatch; deps: httpx, typer, rich; dev: pytest, respx
  README.md
  omnilingual/           # package (modules per §5)
  tests/
    render/golden/       # golden markdown
  docs/superpowers/specs/
```

Implementation note (2026-09-04): no `tests/fixtures/` directory — WAVs are synthesized by `tests/conftest.make_wav` and API JSON is inlined in the tests. Tests otherwise mirror the package layout (`tests/audio/`, `tests/stt/`, `tests/translate/`, `tests/render/`).

## 12. Future extensions (not designed here)

- Input adapter for live system-audio capture on macOS.
- `Translator` implementation using Saaras `mode=translate` for languages Mayura lacks.
- Notion export via public API (`pages.create` / `blocks.append`).
- LLM summary / action items over `transcript.en.md`.
- Speaker diarization (Sarvam offers it at ₹45/hour on batch).
