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
| `--api-key KEY` | Sarvam key, overriding `SARVAM_API_KEY`. Prefer the environment variable: a key passed as a flag lands in your shell history. |
| `--max-chunk-s N` | Longest chunk sent to the API, default 28. Must be under 30 (the synchronous endpoint's limit). |
| `--min-chunk-s N` | Shortest chunk the splitter will produce, default 5. `--max-chunk-s` must be at least twice this. |
| `-v`, `--verbose` | Debug logging, including per-chunk language warnings and cache decisions. |

Exit codes: `0` success, `2` transcript written but some segments failed (see notes in the file), `1` fatal. Usage errors — a missing recording, an unknown or malformed flag — exit `2` as well, since that is Click's convention; those print a usage message rather than a transcript. ffmpeg failures on a corrupt recording print `error: ffmpeg failed (exit N): <stderr>` and exit 1.

## Cost and resumability

Sarvam bills about ₹30 per hour of audio for speech-to-text and ₹20 per 10,000
characters for translation. Every chunk's API result is cached on disk, so
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

## Development

```bash
uv run pytest -q                       # unit + integration (mocked HTTP)
SARVAM_API_KEY=... uv run pytest -m live   # two real API calls, costs a few paise
```

Design: `docs/superpowers/specs/2026-09-04-omnilingual-transcriber-design.md`
