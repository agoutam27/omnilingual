# omnilingual

Transcribe a Zoom local recording of a meeting held in several Indian languages
into a Markdown transcript, each segment in its original language followed by an
English translation. Both halves are pluggable — speech-to-text runs on Sarvam
Saaras (default), local Whisper (MLX on Apple Silicon, faster-whisper on
Intel/Linux CPUs), or Groq's hosted Whisper, and English translation runs on
Sarvam Mayura (default) or Google's Gemini free tier. Every chunk is
auto-detected for language.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- ffmpeg: `brew install ffmpeg`
- At least one API key, unless you run translation locally:
  - `SARVAM_API_KEY` from https://dashboard.sarvam.ai — the default on both axes
  - `GROQ_API_KEY` from https://console.groq.com — free tier, no card
  - `GEMINI_API_KEY` from https://aistudio.google.com/apikey — free tier, no card

  You do not need all three: `--stt groq --mt gemini` runs entirely on free
  tiers and needs no Sarvam key at all. To drop the translation key as well,
  use `--mt indictrans2`, which runs locally (see
  [Install](#install)). Speech-to-text always needs a key or a local model.
  See [Speech-to-text providers](#speech-to-text-providers) and
  [Translation providers](#translation-providers).

## Install

```bash
uv sync
export SARVAM_API_KEY=your-key
```

Optional extras, added on demand:

```bash
uv sync --extra local-stt    # local Whisper (Apple Silicon or Intel CPU)
uv sync --extra diarize      # speaker diarization
uv sync --extra local-mt     # local IndicTrans2 translation (--mt indictrans2)
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
| `--api-key KEY` | Sarvam key, overriding `SARVAM_API_KEY`. Prefer the environment variable: a key passed as a flag lands in your shell history. |
| `--stt NAME` | Speech-to-text backend: `sarvam` (default), `mlx-whisper` (local, Apple Silicon), `faster-whisper` (local, Intel/Linux CPU), or `groq` (cheap cloud). Also on `live`. |
| `--stt-model ID` | Override the model id for the chosen backend. Also on `live`. |
| `--mt NAME` | Translation backend: `mayura` (default, Sarvam), `gemini` (Google AI Studio free tier), or `indictrans2` (local, no quota). Also on `live`. |
| `--mt-model ID` | Override the translation model id for the chosen backend. Also on `live`. |
| `--ask` | Answer numbered prompts for the backend, model, diarization, speaker count, and output path instead of passing the flags above. Also on `live`. |
| `--max-chunk-s N` | Longest chunk sent to the API, default 28. Must be under 30 (the synchronous endpoint's limit). |
| `--min-chunk-s N` | Shortest chunk the splitter will produce, default 5. `--max-chunk-s` must be at least twice this. |
| `-v`, `--verbose` | Debug logging, including per-chunk language warnings and cache decisions. |

Exit codes: `0` success, `2` transcript written but some segments failed (see notes in the file), `1` fatal. Usage errors — a missing recording, an unknown or malformed flag — exit `2` as well, since that is Click's convention; those print a usage message rather than a transcript. ffmpeg failures on a corrupt recording print `error: ffmpeg failed (exit N): <stderr>` and exit 1.

## Cost and resumability

Sarvam bills about ₹30 per hour of audio for speech-to-text and ₹20 per 10,000
characters for translation; `--stt mlx-whisper` makes the speech-to-text half
free, `--stt groq` cuts it to roughly ₹3.4 per hour, and `--mt gemini` drops
translation to zero inside Google's free quota. A run with no Sarvam key at all
(`--stt groq --mt gemini`) costs nothing but the two free tiers' rate limits.
Every chunk's API
result is cached on disk, so
re-running the same command on the same file makes no new API calls. If credits run
out mid-way, fix credits and re-run; only the remaining chunks are sent.

A failure that looks temporary — a 429, a 5xx, a dropped connection, all after
their retries — is not written to the cache: that segment is marked failed in this
run's transcript and retried the next time you run the same command. A failure the
API is clear about, such as a rejected chunk, is remembered, so re-running does not
pay to be told the same thing twice.

## Work directory

Normalized audio, chunks, and the API cache live in
`<out dir>/.omnilingual/<hash of the recording>/` (override with `--work-dir`).
It holds the recording re-encoded as 16 kHz mono 16-bit WAV plus a second copy of
the same audio split into chunks, which comes to about 4 MB per minute of audio —
roughly 350 MB for a 90-minute meeting.

Nothing is cleaned up automatically. Delete the directory to start completely
fresh; that also discards the cache, so the next run pays for the whole recording
again. Because the location is derived from `--out`, changing `--out` points the
tool at a different work directory and abandons the cache built under the old one.

## Speech-to-text providers

| Provider | Cost | Setup | Notes |
|---|---|---|---|
| `sarvam` (default) | ~₹30/hr | `SARVAM_API_KEY` | Saaras, trained on Indian audio including Hindi-English code-mix. |
| `mlx-whisper` | free | `uv sync --extra local-stt` (Apple Silicon only) | Runs `mlx-community/whisper-large-v3-turbo` on-device; the first run downloads ~1.5 GB. Audio never leaves the machine. |
| `faster-whisper` | free | `uv sync --extra local-stt` (Intel Macs and Linux) | Runs Whisper `small` by default (override with `--stt-model`) on CPU via CTranslate2 int8; the first run downloads ~500 MB. The local option for Intel Macs, where MLX is unavailable. |
| `groq` | ~₹3.4/hr ($0.04/hr) | `GROQ_API_KEY` from https://console.groq.com | Hosted `whisper-large-v3-turbo`, much faster than real time. Groq bills a 10 s minimum per request, so short live chunks cost slightly more than the headline rate. |

The same `local-stt` extra installs whichever local engine your platform
supports — MLX on Apple Silicon, faster-whisper everywhere else (on Intel Macs
it pins `onnxruntime<1.24`, the last release with an x86_64 wheel).

All four auto-detect the language of every chunk, so multi-language meetings
work unchanged, and detected codes are normalized to Sarvam's `xx-IN` set,
which is what the translator and the `--langs` filter expect. One caveat: Groq
does not report a detection confidence, so its transcripts show `0.00` where
the others show a probability.

The local and Groq backends are newer than the default; before switching a
workflow over, compare them on your own audio:

```bash
uv run omnilingual meeting.m4a --estimate    # prepares chunks without API calls
uv run scripts/eval_stt.py .omnilingual/<hash>/chunks/0007.wav
```

Translation to English always runs through a Sarvam or Google backend — see
[Translation providers](#translation-providers).

## Translation providers

| Provider | Cost | Setup | Notes |
|---|---|---|---|
| `mayura` (default) | ~₹20/10k chars | `SARVAM_API_KEY` | Sarvam `mayura:v1`, tuned for Indian languages and Hindi-English code-mix. Covers 11 languages; `--mt-model sarvam-translate:v1` widens that to all 22. |
| `gemini` | free | `GEMINI_API_KEY` from https://aistudio.google.com/apikey | `gemini-3.5-flash` via the `generateContent` API. Handles all 22 scheduled languages plus English, and tolerates code-mixed input well. Rate-limited per project. |
| `indictrans2` | free | `uv sync --extra local-mt` | Runs locally on CPU via CTranslate2. **No API quota at all**, so it cannot be rate-limited. Indic→English only, and quality varies sharply by language — reliable for Hindi/Urdu, unreliable for several others. ~850 MB model, downloaded once on first use. |

All three translate to English only, and all three accept every `xx-IN` tag the
speech-to-text providers emit, so the axes combine freely.

### Local translation with `--mt indictrans2`

```bash
uv sync --extra local-mt
uv run omnilingual meeting.m4a --stt groq --mt indictrans2
```

The default weights are the community conversion
[`adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M`](https://huggingface.co/adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M)
(MIT, ungated — no `huggingface-cli login` needed), which packages an
IndicTrans2 checkpoint for CTranslate2 so no PyTorch install or conversion step
is required. Weights are quantized to int8 at load time.

Two caveats worth knowing before you rely on it. It is an **unofficial
conversion** — nobody at AI4Bharat vouches for it, though its structure,
vocabulary, and output are sane. And quality is **very uneven by language**.
Measured on the default weights, Hindi and Urdu translate well; Bengali comes
out partly romanized and partly garbled; and Tamil, Telugu, Gujarati, and Odia
return wrong output (often untranslated text or an unrelated English phrase).
The model is quantized to int8 at load, and a handful of outputs carry a stray
`<unk>`.

Treat it as a good **quota-free default for Hindi-family audio only**, not a
general replacement for `--mt gemini`. If you need the other languages, keep
using Gemini and accept its quota, or point `--mt-model` at a different
repository if you find better weights.

### Choosing a configuration interactively

If you would rather not track the flags, `--ask` walks you through the choices:

```bash
uv run omnilingual live --ask
```

It asks for the speech-to-text backend and model, the translation backend and
model, whether to label speakers (and how many), and the output path. Each
answer accepts either the number or the name, and pressing Enter keeps the
current default — so you can answer everything with Enter to reproduce exactly
what the equivalent flags would have done.

### Running entirely on free tiers

```bash
export GROQ_API_KEY=...      # https://console.groq.com — free, no card
export GEMINI_API_KEY=...    # https://aistudio.google.com/apikey — free, no card
uv run omnilingual meeting.m4a --stt groq --mt gemini
```

No Sarvam account needed. The tool only demands the key for a backend you
actually selected, so this run asks for neither `SARVAM_API_KEY` nor any paid
tier.

Both free tiers are rate-limited rather than unlimited, and in practice this
pairing does exhaust: Groq allows about **8 hours of audio per day** on
`whisper-large-v3-turbo` (20 requests/min, 2,000 requests/day), and Gemini
enforces per-minute and per-day quotas against your AI Studio project — a
multi-language run of a few minutes' audio is enough to trip them, since each
chunk is a separate request. When Gemini returns `429`, that chunk is recorded
as `translation failed` and left untranslated rather than dropped, so a later
run redoes only the failures:

```bash
uv run omnilingual meeting.m4a --stt groq --mt gemini --from-chunks <session dir>
```

If you would rather not think about quotas at all, use
[`--mt indictrans2`](#local-translation-with---mt-indictrans2): it runs on your
own CPU and has no per-project ceiling, so only the speech-to-text provider
spends a network quota.

**Privacy caveat.** Google's free tier states that submitted content
**may be used to improve Google's products**; the paid tier does not. Meeting
transcripts are often confidential, so treat `--mt gemini` as appropriate for
non-sensitive audio only, or use `--mt mayura` with a Sarvam key instead.
Groq's free tier does not carry this caveat, and `--mt indictrans2` never
leaves your machine.

## Speaker diarization

Add `--diarize` to label every transcript segment with its speaker —
`Speaker 1`, `Speaker 2`, … in first-appearance order:

```bash
uv sync --extra diarize            # sherpa-onnx + soundfile; no API key needed
uv run omnilingual meeting.m4a --diarize
uv run omnilingual meeting.m4a --diarize --speakers 3   # when headcount is known
uv run omnilingual live --out live.md --diarize --stt-workers 1
```

Diarization runs fully on-device via sherpa-onnx: the recording is diarized
once and the speaker turns are cached in the work dir, so re-runs are free.
The same embedding models also back live mode, where each sealed chunk is
matched against running speaker centroids (best-effort labels while recording;
`--from-chunks` recovery re-diarizes the saved session for authoritative
labels). Segment headers render `**[00:01:23 → 00:01:31] Speaker 2 · hi-IN**`,
the English-only file prefixes `Speaker 2: `, and the transcript header gains
a per-speaker share line. Labels are anonymous — no cross-recording
voiceprints, no names.

Caveats: a chunk containing two voices is attributed to its dominant speaker;
overlapped speech is not separated. The models first download (~40 MB) to
`~/.cache/omnilingual/models/`. Accuracy on Indian-accented speech is still
under evaluation — spot-check with `uv run scripts/eval_diarize.py meeting.m4a`
before trusting labels on an important meeting.

## Development

```bash
uv run pytest -q                              # unit + integration (mocked HTTP); real-model suites auto-skip
SARVAM_API_KEY=... uv run pytest --run-live   # two real Sarvam API calls, costs a few paise
uv run pytest --run-mlx                       # real local Whisper via MLX (Apple Silicon)
uv run pytest --run-faster-whisper            # real local Whisper on CPU (downloads ~500 MB)
uv run pytest --run-diarize                   # real sherpa-onnx diarization (downloads ~40 MB)
```

Design: `docs/superpowers/specs/2026-09-04-omnilingual-transcriber-design.md`
