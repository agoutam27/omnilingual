# omnilingual

Transcribe a Zoom local recording of a meeting held in several Indian languages
into a Markdown transcript, each segment in its original language followed by an
English translation. Speech-to-text is pluggable — Sarvam Saaras (default),
local Whisper (MLX on Apple Silicon, faster-whisper on Intel/Linux CPUs), or
Groq's hosted Whisper — all with per-chunk language auto-detect; English
translation stays on Sarvam Mayura.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- ffmpeg: `brew install ffmpeg`
- A Sarvam AI API key: https://dashboard.sarvam.ai (needed for translation in
  every setup; speech-to-text has free/cheap alternatives — see
  [Speech-to-text providers](#speech-to-text-providers))

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
| `--api-key KEY` | Sarvam key, overriding `SARVAM_API_KEY`. Prefer the environment variable: a key passed as a flag lands in your shell history. |
| `--stt NAME` | Speech-to-text backend: `sarvam` (default), `mlx-whisper` (local, Apple Silicon), `faster-whisper` (local, Intel/Linux CPU), or `groq` (cheap cloud). Also on `live`. |
| `--stt-model ID` | Override the model id for the chosen backend. Also on `live`. |
| `--max-chunk-s N` | Longest chunk sent to the API, default 28. Must be under 30 (the synchronous endpoint's limit). |
| `--min-chunk-s N` | Shortest chunk the splitter will produce, default 5. `--max-chunk-s` must be at least twice this. |
| `-v`, `--verbose` | Debug logging, including per-chunk language warnings and cache decisions. |

Exit codes: `0` success, `2` transcript written but some segments failed (see notes in the file), `1` fatal. Usage errors — a missing recording, an unknown or malformed flag — exit `2` as well, since that is Click's convention; those print a usage message rather than a transcript. ffmpeg failures on a corrupt recording print `error: ffmpeg failed (exit N): <stderr>` and exit 1.

## Cost and resumability

Sarvam bills about ₹30 per hour of audio for speech-to-text and ₹20 per 10,000
characters for translation; `--stt mlx-whisper` makes the speech-to-text half
free, and `--stt groq` cuts it to roughly ₹3.4 per hour. Every chunk's API
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

Translation to English always goes through Sarvam Mayura, so `SARVAM_API_KEY`
is required regardless of provider.

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
