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

Exit codes: `0` success, `2` transcript written but some segments failed (see notes in the file), `1` fatal. ffmpeg failures on a corrupt recording print `error: ffmpeg failed (exit N): <stderr>` and exit 1.

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
