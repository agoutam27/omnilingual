# STT Provider Alternatives to Sarvam — Research & Integration Design

Date: 2026-09-19
Status: Research complete, design proposed — not yet implemented
Audience: omnilingual maintainers

## 1. Context & Goal

omnilingual currently depends on a single vendor, Sarvam AI, for speech-to-text
(`saaras:v4` via `POST /speech-to-text`) and machine translation (`mayura:v1` via
`POST /translate`). This document:

1. surveys open-source / free / cheap voice-to-text models for Indian languages,
2. evaluates them against omnilingual's actual architectural requirements, and
3. lays out an end-to-end, phased integration plan.

Scope: **STT only.** Translation (Mayura) stays as-is; a local-MT follow-up
(IndicTrans2) is noted as future work in §8.

## 2. Current Integration (verified file:line map)

The codebase already has a provider seam — this plan plugs into it rather than
inventing a new one.

| Piece | Location | Notes |
|---|---|---|
| `STTProvider` Protocol | `omnilingual/stt/base.py` | `model: str`, `mode: str`, `transcribe(wav_path: Path) -> STTResult` |
| `Translator` Protocol | `omnilingual/translate/base.py` | `model`, `supports(lang)`, `to_english(text, src_lang)` |
| Sarvam STT impl | `omnilingual/stt/sarvam.py` (52 lines) | multipart WAV POST to `{base_url}/speech-to-text`, header `api-subscription-key`, `data={model, mode:'transcribe', language_code:'unknown'}`; parses `language_code`, `language_probability`, `transcript` into `STTResult` |
| Provider construction | `omnilingual/cli.py` — **3 hardcoded sites** | `_run_from_chunks` (~L109), `transcribe` (~L196), `live` (~L306) all call `SarvamSTT(settings), MayuraTranslator(settings)` directly |
| Settings | `omnilingual/config.py` | frozen dataclass; `require_key()` always demands `SARVAM_API_KEY`; rates `stt_inr_per_hour=30.0`, `mt_inr_per_10k_chars=20.0` |
| Cache | `omnilingual/cache.py` | `stt_key(wav_bytes, model, mode)` — **a new provider gets isolated cache entries for free** via a distinct `model` string; no cache changes needed |
| Error tree | `omnilingual/http.py` | `SarvamError → AuthError(401/403) / QuotaError(402) / TransientError(429,5xx,network)`; `send_with_retry` backoff+jitter. Generic HTTP semantics — reusable by any REST provider |
| Audio contract | `omnilingual/audio/`, `pipeline/__init__.py:prepare()` | 16 kHz mono 16-bit WAV, chunks 5–28 s |
| 30 s ceiling | `omnilingual/cli.py:57` | `MAX_CHUNK_LIMIT_S = 30.0` exists **only** because Sarvam's sync endpoint rejects ≥30 s audio |
| Live mode | `omnilingual/pipeline/live.py` | `stt_workers=2` worker threads call `stt.transcribe` **concurrently** → any local provider must be thread-safe or serialized |
| Dependency policy | `pyproject.toml` | runtime deps are exactly `httpx`, `typer`, `rich` — heavyweight ML deps are a real design cost |

Behavioral requirements any alternative must preserve:

- **R1 Per-chunk language auto-detect with probability** — the pipeline sends
  `language_code='unknown'` and uses the returned code/confidence per chunk;
  mixed Hindi/Tamil/English meetings are the core use case.
- **R2 BCP-47 `-IN` language codes** (`hi-IN`, `en-IN`…) — `en-IN` skips MT;
  `MayuraTranslator.supports()` matches `-IN`-suffixed codes.
- **R3 `STTResult(lang, prob, text)`** shape and empty-text → `no_speech`.
- **R4 Cache compatibility** — distinct `model`/`mode` strings per provider.
- **R5 Thread-safety** under `live --stt-workers 2`.
- **R6 Minimal core deps** — local-ML deps must be an optional extra.
- **R7 Mac-first** — capture is AVFoundation; primary runtime must run well on
  Apple Silicon.

## 3. Research: Model Landscape

### 3.1 Open-source models (verified on Hugging Face / papers)

| Model | Params | License | Indic coverage | Hindi WER (benchmark) | Per-chunk LID | Inference stack |
|---|---|---|---|---|---|---|
| **Whisper large-v3 / large-v3-turbo** (OpenAI) | 1.55B / 809M | MIT | 99+ langs incl. hi bn ta te ml mr gu kn pa or ur as… | turbo ≈ large−~2% (OpenAI claim); zero-shot Indic numbers **not benchmarked here** — see risk §7 | **Yes, native, with probability** | mlx-whisper, whisper.cpp, faster-whisper, transformers |
| **IndicWhisper** (AI4Bharat, Vistaar/Interspeech 2023) | 769M (whisper-medium FT) | MIT | 12 langs, **one checkpoint per language** | Vistaar Hindi avg **13.6** (vs Azure 20.0, Google 23.9, Nvidia-large 18.6); Kathbath 10.3 | **No** — language pinned via forced decoder ids | plain `transformers` pipeline |
| **IndicConformer-600M-Multi** (AI4Bharat) | 600M | MIT | **all 22 scheduled languages** | Vaani Hindi **13.2** | **No** — needs separate LID stage | **NVIDIA NeMo** (heavyweight; torch + NeMo toolkit) |
| **whisper-large-v3-vaani-hindi** (ARTPARK-IISc) | 1.55B (large-v3 FT, ~718 h Hindi) | MIT (check per-repo) | Hindi (+ a collection of 11 per-language Vaani fine-tunes) | **Kathbath 8.85**, FLEURS 11.2, CommonVoice 13.84, Vaani 24.66 | Inherits Whisper LID (mono-language fine-tune — trust LID only within `--langs`) | mlx-whisper / transformers (it's a Whisper checkpoint) |
| Meta MMS-1B-all / SeamlessM4T v2 | 1B / 2.3B | **CC-BY-NC 4.0** | 1000+ / ~100 langs | competitive | Yes (MMS LID) | transformers — **non-commercial license: disqualified for default** |
| Vosk small Indic models | ~50M | Apache-2.0 | hi, gu, te, ta… | far above Whisper-class | partial | vosk — quality too low for meetings |

Key Vistaar context: across 59 benchmarks × 12 languages, IndicWhisper is best
in 39. But its one-checkpoint-per-language shape conflicts with R1 (per-chunk
auto-detect), as does IndicConformer's need for a separate LID model.

### 3.2 Apple Silicon runtimes for Whisper-family

| Runtime | License | Mac backend | large-v3 speed (M-series) | Notes |
|---|---|---|---|---|
| **mlx-whisper** (ml-explore) | MIT | Metal GPU via MLX | fastest in mac-whisper-speedtest relative ranking (turbo: 1.02 vs whisper.cpp-CoreML 1.23 vs faster-whisper-CPU 6.96); greedy decoding only | **pure `pip install mlx-whisper`** — no binary build; fits R6 as an optional extra |
| whisper.cpp | MIT | Metal / CoreML ANE | M2 Metal RTF ≈ 0.45 (30 s chunk ≈ 13.5 s); turbo 14–18× RT on M5 | needs brew/binary or immature `pywhispercpp`; strong fallback if mlx misbehaves |
| faster-whisper | MIT | **no Metal — CPU-only on Mac** (~3× RT large-v3 int8); ~12× RT on NVIDIA | built-in Silero VAD + LID with probability | best choice for a future Linux/CUDA port, not for this Mac-first repo |

### 3.3 Hosted cheap APIs (June-2026 pricing, from pricing pages)

| Provider | Model | $/audio-hour | Indic langs | Auto-LID/req | Notes |
|---|---|---|---|---|---|
| **Groq** | whisper-large-v3-turbo | **$0.04** (batch $0.02) | 99+ (Whisper coverage) | language in verbose_json | OpenAI-compatible REST; ~216–300× RT; pads <30 s→30 s, min charge 10 s/req; free tier |
| Groq | whisper-large-v3 | $0.111 | 99+ | same | quality tier |
| OpenAI | gpt-4o-mini-transcribe | $0.18 | good | limited verbose output | launched Mar-2025 |
| Google | Chirp 3 dynamic batch | $0.24 | strong | yes | 60 min/month free |
| Deepgram | Nova-3 multilingual | $0.31 | narrower Indic set | yes | $200 credits |
| **Sarvam (baseline)** | saaras:v4 | **≈$0.36 (₹30)** | 11–23, best code-mix tuning | yes | current default |
| Local mlx-whisper | turbo | **₹0 marginal** | 99+ | yes | offline, private |

## 4. Decision

Three tiers, one protocol:

1. **Default stays Sarvam** until the §6.4 A/B eval on real meeting recordings
   shows local quality is acceptable on code-mixed audio (Sarvam is tuned for
   exactly this; zero-shot Whisper is the known risk).
2. **`MlxWhisperSTT` — primary alternative (local, free, offline).**
   `mlx-whisper` + `mlx-community/whisper-large-v3-turbo` (default) or
   `…-large-v3` (quality). Whisper's native LID satisfies R1/R3; an
   ISO-639-1→`-IN` mapping table satisfies R2; a `threading.Lock` satisfies R5;
   distinct `model` string satisfies R4; optional extra satisfies R6.
   The `model` setting accepts any HF/MLX whisper repo id → ARTPARK Vaani
   fine-tunes (Kathbath 8.85 Hindi) become a config-level quality upgrade when
   `--langs` is narrow.
3. **`GroqSTT` — cheap cloud fallback.** OpenAI-compatible multipart endpoint,
   reuses `http.py` retry/error machinery; turbo ≈ ₹3.4/hr vs Sarvam ₹30/hr.
   For users without a Sarvam key or with big backlogs.

Deferred with reasons: IndicConformer (NeMo dep weight + no LID stage),
IndicWhisper (no LID; 12 separate checkpoints), MMS/SeamlessM4T (CC-BY-NC),
Vosk (quality), faster-whisper (no Metal), local MT via IndicTrans2 (separate
scope).

## 5. Design

### 5.1 New/changed files

| File | Change |
|---|---|
| `omnilingual/stt/mlx_whisper.py` *(new)* | `MlxWhisperSTT(STTProvider)`: lazy-loads model on first `transcribe`; `mode='transcribe'`; `model=f"mlx-whisper:{repo_id}"`; `threading.Lock` around inference; lang mapping; prob via `mlx_whisper` decode info (fallback: explicit `detect_language` pass, else `0.0` with a code comment) |
| `omnilingual/stt/groq.py` *(new)* | `GroqSTT(STTProvider)`: httpx multipart POST to `https://api.groq.com/openai/v1/audio/transcriptions`, `Authorization: Bearer`, `response_format=verbose_json`; reuses `send_with_retry`; maps HTTP statuses onto the existing error tree; `model=f"groq:{repo_id}"`; lang mapping shared with mlx provider |
| `omnilingual/stt/langs.py` *(new)* | `WHISPER_TO_BCP47 = {"hi":"hi-IN", "ta":"ta-IN", "te":"te-IN", "kn":"kn-IN", "ml":"ml-IN", "mr":"mr-IN", "gu":"gu-IN", "bn":"bn-IN", "pa":"pa-IN", "or":"or-IN", "as":"as-IN", "ur":"ur", "ne":"ne", "en":"en-IN", …}` — pass-through for unmapped codes |
| `omnilingual/stt/__init__.py` | `build_stt(settings) -> STTProvider` factory; raises `ConfigError` with install hint (`uv pip install 'omnilingual[local-stt]'`) if mlx import fails |
| `omnilingual/config.py` | `Settings` gains `stt_provider: str = "sarvam"` and `groq_api_key: str = ""` (+ `GROQ_API_KEY` env); `stt_model` becomes provider-scoped default (sarvam→`saaras:v4`, mlx→`mlx-community/whisper-large-v3-turbo`, groq→`whisper-large-v3-turbo`); provider-carried `inr_per_hour` (sarvam 30, groq ≈3.4, local 0) read by `pipeline._price` via `getattr(stt, "inr_per_hour", settings.stt_inr_per_hour)` |
| `omnilingual/cli.py` | `--stt {sarvam,mlx-whisper,groq}` and `--stt-model` on `transcribe` and `live`; the 3 hardcoded construction sites → `build_stt(settings)`; keep `MAX_CHUNK_LIMIT_S=30.0` (Whisper's native window — constraint stays valid); key check stays tied to Mayura (MT still Sarvam) |
| `pyproject.toml` | `[project.optional-dependencies] local-stt = ["mlx-whisper>=0.4", "mlx"]`; core deps untouched |
| `README.md`, `scripts/setup-mac.sh` | document providers, costs, first-run model download (~1.5 GB turbo), `--stt-workers 1` guidance for local live mode |

### 5.2 Behavior notes

- **Empty/silent chunks**: Whisper may emit whitespace or hallucinated tokens on
  silence; keep the existing empty-text → `no_speech` path, and add a guard:
  strip + collapse known Whisper silence artifacts (`""`, `"."`, `"..."`,
  `"Thank you."` on <2 s chunks) before classification — *flagged for the
  §6.4 eval to validate, not assumed*.
- **Cost display**: `Cost.inr_estimate` becomes provider-aware via the
  `inr_per_hour` attribute; local runs display ₹0 STT + real MT cost.
- **Live mode**: local provider serializes inference (lock); document
  `--stt-workers 1`. 8 s chunks at turbo speed (≈0.5–2 s each on M3) keep up
  easily.
- **`prob` fidelity**: Sarvam and Whisper LID both give real probabilities;
  Groq `verbose_json` returns `language` without a probability → `prob=0.0`
  and the renderer already tolerates low values (display-only field).

## 6. End-to-End Implementation Plan

Each phase is independently mergeable; Phases 1–3 are code, Phase 4 is the
go/no-go for changing the default.

### Phase 1 — Factory & plumbing (no behavior change)
1. `stt/__init__.py`: add `build_stt(settings)` returning `SarvamSTT(settings)` for now.
2. `cli.py`: replace the 3 hardcoded sites with `build_stt(settings)`.
3. `config.py`: add `stt_provider`, `groq_api_key`, env parsing, validation.
4. Tests: factory returns SarvamSTT by default; existing suite stays green.

### Phase 2 — Local Whisper provider
1. `pyproject.toml`: `local-stt` extra; `stt/langs.py` mapping table.
2. `stt/mlx_whisper.py` per §5.1 (lazy load, lock, mapping, prob strategy).
3. `--stt/--stt-model` CLI flags wired through `Settings`.
4. Tests (unit, no model download): mapping table; factory error message when
   mlx missing; lock serialization with a fake transcribe; cache-key isolation
   (`stt_key` differs from sarvam's for identical wav bytes).
5. Tests (marked `mlx`, skipped without the extra): 1 s synthetic-tone wav →
   `STTResult` shape; a bundled 5 s Hindi clip fixture → `lang == "hi-IN"`,
   `prob > 0`.
6. `pipeline._price` provider-rate change + tests.

### Phase 3 — Groq provider
1. `stt/groq.py` per §5.1 (verbose_json, error mapping, shared lang table).
2. Tests with `respx`: happy path (language/text parsed, `prob=0.0`), 401→AuthError,
   429→TransientError with retry, multipart body shape.
3. README cost table update.

### Phase 4 — Evaluation & default decision
1. `scripts/eval_stt.py` (manual, not CI): transcribe 2–3 real archived meeting
   recordings with `--stt sarvam` and `--stt mlx-whisper`; diff segment counts,
   language distribution, and spot-check 20 random segments per provider for
   transcript quality (human pass).
2. Acceptance bar: no systematic language misdetection; subjective transcript
   quality "usable for meeting notes" on code-mixed audio; per-chunk latency
   ≤ 3 s for 8 s chunks on the maintainer's Mac.
3. If passed: flip `stt_provider` default to `mlx-whisper` in a separate PR;
   Sarvam remains `--stt sarvam`. If failed: keep default, record findings in
   this doc.

## 7. Risks & Open Questions

- **R-1 Zero-shot Whisper quality on code-mixed Hinglish** — Sarvam trains for
  this exactly; unverified for turbo. Mitigated by Phase 4 gate.
- **R-2 mlx-whisper language-probability exposure** — high-level
  `transcribe()` returns `language`; probability may require the lower-level
  `detect_language` pass. Fallback `prob=0.0` is acceptable (display-only) but
  Phase 2 should confirm the real API.
- **R-3 Whisper silence hallucinations** on 5 s quiet chunks — guard in §5.2;
  validate in Phase 4.
- **R-4 First-run 1.5 GB model download** — document; consider a
  `scripts/setup-mac.sh` prefetch step.
- **R-5 Groq billing quirks** (10 s minimum, <30 s padding) — live-mode 8 s
  chunks billed as 10 s; reflected in the provider's `inr_per_hour` only
  approximately; noted in README.

## 8. Future Work (explicitly out of scope)

- Local MT via AI4Bharat IndicTrans2 (would make `SARVAM_API_KEY` fully
  optional; torch/transformers weight — separate design doc).
- IndicConformer via NeMo + AI4Bharat LID stage (revisit if IndicConformer
  ships a lighter ONNX path).
- faster-whisper provider for Linux/CUDA portability.
- whisper.cpp brew-binary provider if mlx-whisper proves unstable.

## 9. Sources

- AI4Bharat IndicConformer: huggingface.co/ai4bharat/indic-conformer-600m-multilingual (MIT, Vaani Hindi WER 13.2)
- Vistaar / IndicWhisper: arXiv:2305.15386 (Interspeech 2023), AI4Bharat HF checkpoints (MIT)
- ARTPARK-IISc Vaani fine-tunes: huggingface.co/ARTPARK-IISc/whisper-large-v3-vaani-hindi (Kathbath 8.85)
- Prompt-tuning Whisper for Indic: arXiv:2412.19785 (Kathbath hi 9.24)
- Groq pricing: console.groq.com/docs/pricing (turbo $0.04/hr, batch $0.02/hr)
- OpenAI pricing: openai.com/api/pricing (gpt-4o-mini-transcribe $0.003/min)
- Deepgram / AssemblyAI / Google Chirp pricing pages (June 2026 captures)
- Runtime benchmarks: github.com/whispercpp benchmark tables, SpeakUp M2 RTF measurements, mac-whisper-speedtest relative rankings, ml-explore/mlx-examples (mlx-whisper)
- Repo internals: `omnilingual/{stt,translate,pipeline,cli.py,config.py,cache.py,http.py,models.py}` (read 2026-09-19)

## 10. Implementation Status (2026-09-19)

Phases 1–3 are implemented and tested (216 passed, 4 skipped):

- Phase 1: `build_stt(settings)` factory in `omnilingual/stt/__init__.py`;
  `stt_provider` / `stt_model` / `groq_api_key` on `Settings` with
  `resolved_stt_model` per-provider defaults; `--stt` / `--stt-model` on
  `transcribe` and `live`; provider-aware pricing via an optional
  `inr_per_hour` provider attribute (Sarvam falls back to the configured rate).
- Phase 2: `omnilingual/stt/mlx_whisper.py` — `MlxWhisperSTT` with lazy engine,
  an explicit `detect_language` pass (R-2 resolved: real probability), and a
  thread lock for live mode's concurrent workers; `omnilingual/stt/langs.py`
  Whisper→BCP-47 map (incl. Whisper `or` → Sarvam `od-IN`).
- Phase 3: `omnilingual/stt/groq.py` — `GroqSTT` on the OpenAI-compatible
  endpoint reusing `send_with_retry` and the `SarvamError` tree; `GROQ_API_KEY`
  fail-fast. Groq exposes no language probability, so `prob=0.0` there.
- Also new: `scripts/eval_stt.py` A/B harness, `local-stt` optional extra gated
  to macOS arm64, `mlx` pytest marker, README provider section.

Phase 4 (flipping the default provider) remains gated on the manual eval in §6:
run `scripts/eval_stt.py` on ≥10 chunks per expected meeting language, then one
full meeting with `--stt mlx-whisper`.
