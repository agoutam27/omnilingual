# AGENTS.md — omnilingual

Multilingual Indian-language meeting audio → Markdown transcript: each segment in its original language plus an English translation. Three entry points: batch file (Zoom recording), live macOS capture, and a local web UI. Python 3.12, `uv`, `ffmpeg` required.

## Run
```bash
uv sync [--extra local-stt --extra diarize ...]          # list ALL wanted extras in one call: uv sync removes unlisted ones
uv run omnilingual REC.m4a [--stt groq --mt gemini --diarize --estimate]   # anything not a subcommand routes to `transcribe`
uv run omnilingual live --out live.md [--ask]            # needs "Omnilingual" aggregate device (BlackHole + mic)
uv run omnilingual --from-chunks .omnilingual/live-<ts>  # finish/recover a halted live session
uv run omnilingual-ui [--no-window]                      # FastAPI on loopback + pywebview window
uv run pytest -q                                          # ~570 tests, mocked HTTP; real-model suites opt-in:
#   --run-live (Sarvam, costs paise) --run-mlx --run-faster-whisper --run-indic-transcribe --run-diarize
./scripts/setup-mac.sh [--check|--dry-run]               # idempotent machine setup + host due-diligence (RAM/disk/network/HF access); --check changes nothing
```
Keys (only for selected backends): `SARVAM_API_KEY` (sarvam/mayura), `GROQ_API_KEY`, `GEMINI_API_KEY`. CLI reads process env only (`uv run --env-file .env …`); UI reads repo `.env` itself. Exit codes: 0 ok, 2 written with failed segments, 1 fatal.

## Data flow
- Batch (`pipeline/__init__.py`): `normalize` (ffmpeg → 16 kHz mono s16le WAV) → `chunk_audio` (silencedetect cuts, `min_chunk_s..max_chunk_s`, <30 s Sarvam sync limit) → optional diarize whole file → per chunk `_stt_cached` → `_mt_cached` (skip `en-IN`) → `Transcript` → `render/markdown.py`.
- Live (`pipeline/live.py`): `LiveCapture` (one ffmpeg avfoundation input, PCM stdout + silence events stderr) → `LiveSlicer` seals chunks at silence after `target_s` → worker threads `process_chunk` → `LiveMarkdownWriter` appends in order. Cost cap / quota / auth halt transcription but keep capturing; recover via `--from-chunks` (`run_from_chunks`, re-diarizes authoritatively).
- UI (`ui/`): `server.py` routes + `/ws` event stream → `session.py` `SessionRunner` (the ONLY ui module that imports `omnilingual.pipeline`) runs pipeline on a thread, bridging callbacks (`on_segment`, `on_cost`, `stop_event`) to the asyncio loop.
- Work dirs: batch `<out dir>/.omnilingual/<sha12 of recording>/{normalized.wav,chunks/,chunks.json,chunking.json,cache/}`; live `.omnilingual/live-<utc>/{session.json,live-chunks/,chunks.json,cache/}`.

## Layout
`config.py` Settings, provider/model registries, all validation + `require_keys` · `models.py` Chunk/STTResult/Segment/Transcript dataclasses · `http.py` retry + error types · `cache.py` content-addressed JSON cache · `stt/` `translate/` `diarize/` = `base.py` Protocol + `__init__.py` factory + one module per backend · `audio/` ffmpeg, chunker, live capture/slicer · `render/` batch + live writers · `cli.py` Typer app · `ui/` web app · `scripts/` setup, CoreAudio device tool (Swift), eval scripts · `docs/superpowers/{specs,plans}` design history.

## Invariants (break these and paid results or tests break)
- **Error policy lives only in `pipeline._cached_call`**: `AuthError`/`QuotaError` propagate (abort, resumable); `TransientError` (after 3 retries in `http.send_with_retry`) is NOT cached → retried next run; other `SarvamError` cached as `{"failed": true}`. All provider HTTP goes through `send_with_retry`.
- Cache keys = sha256 of wav bytes+model+mode (STT) / text+src+tgt+model (MT). Changing key shape or `provider.model`/`mode` re-bills every user's cache.
- Never drop a segment: statuses are `ok|no_speech|stt_failed|mt_unsupported|mt_failed`; silence has no speaker. Output format is pinned by `tests/render/golden/*.md`.
- Language codes are Sarvam-style `xx-IN` everywhere (Odia = `od-IN`); Whisper backends normalize via `stt/langs.py`.
- `stt_model`/`mt_model=None` means provider default from `DEFAULT_*_MODELS`; never leak one provider's model id into another. Per-model MT input caps in `MT_CHAR_LIMITS`.
- Pricing: provider may expose `inr_per_hour` / `inr_per_10k_chars` (read via `getattr`); free backends set `0.0`.
- Optional extras are imported lazily inside factories/constructors; CLI must never import `omnilingual.ui` (test-enforced) so core install stays web-free.
- Validation (chunk bounds, keys, speakers) lives in `config.py` and is shared by CLI and UI — never duplicate it.
- UI security: loopback-only bind, launch token + Host/Origin checks; API key values never leave `ui/secrets.py`, never go in argv/logs/events; in UI runs `.env` overrides process env. `secrets.py` mirrors `setup-mac.sh` `.env` semantics — change both together. Extras live in three places that a test keeps equal: `pyproject.toml`, `setup-mac.sh` `ALL_EXTRAS`, `ui/server.py` `_SETUP_EXTRAS`. UI builds providers off the event loop (local providers load their model in `build_stt`). UI prefs: `~/.config/omnilingual/ui.toml`.
- Live: nothing may kill a worker or callback (`on_segment`, `on_cost`) — an escaped exception leaves the ordered appender waiting forever; log it and mark the chunk failed. `--max-cost` caps true spend (`spent`); `accrued` = kept chunks only, so display spend by summing `delta`.
- `run_from_chunks` diarizes the concatenated speech chunks (gaps dropped): match turns against the re-timed spans from `_combine_session_wavs`, never session-clock `chunk.start_s`.
- `pyproject.toml` pins with comments (`sherpa-onnx-core` explicit, `onnxruntime<1.24` on Intel Mac) are deliberate; keep them.

## Adding a backend
Add name + default model to `STT_PROVIDERS`/`MT_PROVIDERS` and `DEFAULT_*_MODELS` in `config.py`, key check in `require_keys`, class satisfying `stt/base.py` (`model`, `mode`, `transcribe(wav)->STTResult`) or `translate/base.py` (`model`, `supports(lang)`, `to_english(text, src)`), branch in the factory, CLI help text, README table, tests in `tests/stt|translate/`.

## Code conventions
- `from __future__ import annotations`; frozen dataclasses; `typing.Protocol` interfaces; module docstring on every file.
- Inject seams for tests (`client: httpx.Client`, `sleep`, `env`, `capture_factory`); mock HTTP with `respx`; WAV fixtures via `tests/conftest.py` `make_wav`. Tests mirror package layout; real-model tests carry markers.
- Comments explain *why* (constraints, rejected alternatives), not what. Match that density.
- Rich output: `escape()` any external text (ffmpeg stderr, API bodies, lang codes).
- Workflow: spec in `docs/superpowers/specs/` → plan in `docs/superpowers/plans/` → TDD. Conventional commits with scope, e.g. `feat(ui): …`, `fix(setup): …`, `docs(plan): …`.
