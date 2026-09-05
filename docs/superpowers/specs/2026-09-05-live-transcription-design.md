# Omnilingual Live: macOS Mic + System Audio → Live Markdown — Design

Date: 2026-09-05
Status: Draft rev 2 (incorporates Oracle adversarial critique; pending approval)
Builds on: `docs/superpowers/specs/2026-09-04-omnilingual-transcriber-design.md` (v1 file pipeline)

## 1. Goal

Extend omnilingual from post-meeting file transcription to live meeting transcription on macOS: listen to microphone + system audio (Zoom/Meet), transcribe-translate on the fly, and grow a Markdown file incrementally so `tail -f` shows the meeting as it happens.

Latency commitment: ~10–15 s typical, ~30 s worst case (continuous uninterrupted speech forces hard cuts at the 28 s ceiling). Not real-time subtitles.

v1-live scope, per clarifications 2026-09-05:
- macOS only, Zoom/Meet live, mic + speaker output mixed to single mono (combined is fine, no speaker tags).
- Reuse the current synchronous chunk pipeline (Saaras STT + Mayura MT + JsonCache), no streaming STT.
- Output is the Markdown file only — no TUI. `Ctrl+C` stops, file stays valid at every moment.

## 2. Context and constraints

- Reuses v1 stages: silence thresholds default `-35 dB / 0.4 s` (live adds `--noise-db` override, §4), chunk bounds `min 5 s / max 28 s` (hard API limit 30 s), `SarvamSTT.transcribe(wav) -> STTResult`, `Translator.to_english`, `JsonCache`, segment rendering.
- macOS exposes no system audio to ffmpeg. v1-live uses BlackHole 2ch behind an Aggregate Device (§4); ScreenCaptureKit is a future capture backend behind the same interface (§5).
- ffmpeg `avfoundation` reads ONE audio device per `-i`, and there is no addressable "system default input" alias — the capture device must be named. Mixing two devices with `amix` is rejected (clock drift, unbounded queues, level renormalization breaks silence thresholds); one coherent device is required.
- Live audio is unbounded: no upfront duration, cost accrues while running. Budget is scarce (v1 §2) → local energy gate skips STT on silence chunks, and `--max-cost` caps spending.
- Markdown must be valid at every Ctrl+C and survive mid-session ffmpeg death; `tail -f` must see every update.
- Python 3.12+ target but `audioop` (removed in 3.13) is not used; RMS computed with `array`/`struct`. No new third-party deps.

## 3. Approach chosen

**A. BlackHole + Aggregate Device + single ffmpeg rolling chunker (v1-live).** One-time setup creates a Multi-Output Device (headphones + BlackHole) so the meeting stays audible AND capturable, and an Aggregate Device (mic + BlackHole) so both sources arrive at ffmpeg as ONE device on one clock. One ffmpeg child outputs 16 kHz mono s16le PCM to stdout while the same process runs `silencedetect` on a side branch (§4). A rolling buffer seals at the first silence after a target age; each sealed WAV reuses the existing `_stt_cached` / `_mt_cached` path and is appended to the `.md`.

Alternatives rejected for v1-live, kept as follow-ups:
- **Two avfoundation inputs + amix**: drifts apart over minutes, queue growth unbounded, and `amix`'s level normalization halves signal energy, silently breaking the -35 dB silence threshold. The Aggregate Device solves all three at the OS level.
- **B. ScreenCaptureKit native (zero-install)**: better UX (no driver), but needs Screen Recording permission flow, a Swift/ObjC bridge, and drift handling — 2–3x work for identical transcript quality. The `LiveCapture` interface (§5) exists so B can replace A later.
- **C. True streaming STT (2–3 s partials)**: contradicts the accepted latency commitment; discards cache/cost model; partial-translation flicker is a separate problem. Revisit only if sub-5 s is ever required.

## 4. Audio topology and sealing rule

### Device setup (one time, §11)

1. Install BlackHole 2ch.
2. **Multi-Output Device** = headphones + BlackHole 2ch. System output → this device. (Headphones are the documented default, not speakers: with speakers, the mic picks up remote voices and every remote utterance is transcribed twice — echo cannot be preflight-detected. Also: macOS volume keys do nothing on a Multi-Output Device; set headphone volume before switching.)
3. **Aggregate Device** = mic + BlackHole 2ch, clock source = BlackHole, drift correction on the mic sub-device. Name it `Omnilingual` (default matched by `--input`).

### Single ffmpeg topology

```
ffmpeg -f avfoundation -thread_queue_size 8 -i ":<aggregate-index>"
  -af "asplit=2[pcm][det];
       [pcm]pan=mono|c0=0.5*c0+0.25*c1+0.25*c2,aresample=16000[out];
       [det]silencedetect=noise=<noise>dB:d=0.4"
  -map "[out]" -c:a pcm_s16le -f s16le -
```

One process, one device, one clock: raw PCM to stdout, `silencedetect` events to stderr parsed with the same regexes as `audio/chunker.py` (exact v1 silence semantics). Explicit `pan` scaling (mic 0.5, BlackHole L/R 0.25 each — initial values, calibrated during implementation against real recordings) keeps a consistent mix with no `amix`-style auto-normalization. Because any rescaling shifts the noise floor, `--noise-db` (default `-35`) is exposed: AGC or a noisy room can raise the floor permanently, in which case no silence is ever detected, every chunk hard-cuts at 28 s, and latency pins at the worst case. Aggregate channel layout (mic + 2ch = 3 channels) is asserted at startup by parsing `-list_devices`/probe output; a different layout aborts with the fix-it hint.

Degraded mode `--mic-only`: single-device capture without BlackHole (useful for testing; remote audio absent).

### Sealing rule (minimizes chunk age)

Buffer PCM from the byte-clock (`session_time = bytes_read / 32000`, not wall clock — wall clock drifts with pipe buffering and macOS sleep). Once buffered audio reaches `target_s` (default 8), seal at the **first** silence-start thereafter (≤ `max_s`); if none arrives by `max_s`, hard-cut at `max_s`. Cutting at silence-start (not midpoint, unlike the file chunker) minimizes chunk age; the gap itself is discarded — it is never transcribed and never billed. `target_s ≥ min_s` structurally. This is the live inverse of the file chunker's "latest silence" rule, which maximizes chunk age.

Local energy gate before STT: compute RMS per sealed chunk; below a speech-energy floor (calibrated in the startup probe) the chunk is marked `no_speech` locally — no API call, no cost, nothing appended unless it closes a >30 s gap.

## 5. Architecture and threading model

```
Aggregate Device (mic + BlackHole)
      │  one ffmpeg child (PCM stdout │ silencedetect stderr)
      ▼
capture thread ──► rolling buffer (spill to disk under backpressure)
      │                    │ stderr reader thread parses silence events
      ▼                    ▼
slicer (seals at first silence ≥ target_s) ─► live-chunks/NNNN.wav + manifest append
      │
      ▼  job queue (chunk idx order)
STT/MT worker pool (--stt-workers, default 2)   [reuses _stt_cached / _mt_cached]
      │
      ▼  results held until all lower idx appended
appender ─► O_APPEND segment write + in-place header update + fsync ─► meeting.md
```

Mandatory separation (dropped audio otherwise): the capture thread drains ffmpeg's stdout unconditionally — a blocking API retry (worst ~7 s backoff) in the same loop would fill the 64 KB kernel pipe, stall ffmpeg, and overflow CoreAudio. If workers fall behind, sealed WAVs queue on disk without bound; lag is logged every 30 s to stderr (`[live 00:14:20] sealed 42 · appended 38 · lag ~11 s`). Lag grows only when STT throughput < real time, which two workers prevents for typical speech density.

`LiveCapture` interface (ScreenCaptureKit-swappable):

```python
class LiveCapture(Protocol):
    def open(self) -> None: ...              # spawn child; CaptureError on device/permission failure
    def __iter__(self) -> Iterator[bytes]: ...  # 16 kHz mono s16le, gapless
    def close(self) -> None: ...             # always reaps the child (used as context manager)
```

Contract: construction takes an unambiguous device name (substring match; multiple matches → error listing candidates); iteration yields PCM and raises `CaptureError` if the capture child dies (clean close → `StopIteration`).

Components:

| Module | Responsibility |
|---|---|
| `audio/live_capture.py` | ffmpeg avfoundation child, device discovery/parse, `--check-audio` support, `CaptureError`. |
| `audio/live_slicer.py` | Rolling buffer, silence-event consumption, sealing rule, energy gate, chunk WAV writes, manifest append. Pure w.r.t. time: byte-clock only. |
| `pipeline/live.py` | `run_live(...)`: wires capture → slicer → worker pool → ordered appender; lag monitor; `--max-cost` enforcement; graceful/partial Ctrl+C. |
| `render/live.py` | Crash-safe incremental writer (§8). Reuses `render/markdown` segment formatting so live and batch bodies are identical. |
| `cli.py` (`live` cmd) | Flags, validation, preflight, status lines to stderr (the file is the product). |

## 6. Data model, work dir, session identity

Reuse `Chunk`, `STTResult`, `Segment`, `Cost`, `Transcript` unchanged from v1 §6 (cache keys and golden tests stay compatible). `start_s`/`end_s` are byte-clock session offsets.

Work dir: `<out dir>/.omnilingual/live-<UTC stamp>/`

```
session.json           # device names, settings, start time — session identity
chunks.json            # manifest, appended per sealed chunk (recovery, §7)
live-chunks/0000.wav … NNNN.wav
cache/stt/<key>.json  cache/mt/<key>.json
```

## 7. Error handling, cost guard, and recovery

| Situation | Behaviour |
|---|---|
| BlackHole/Aggregate/Multi-Output missing | `--check-audio` + startup preflight list avfoundation devices, run a 1 s capture with RMS sanity check per source (catches silent BlackHole / output-not-set — the most common real failure), and print the §11 fix steps. Exit 1 before any API call. |
| Mic permission (TCC) | ffmpeg stderr's AVFoundation permission error → print Settings ▸ Privacy ▸ Microphone path; exit 1. |
| ffmpeg missing | `ensure_ffmpeg()` gate as batch. |
| Echo (remote utterance twice) | Documented: headphones default (§4); troubleshooting row; `--check-audio` reminds. Not preflight-detectable. |
| API 429/5xx | Same retry via `http.py` inside workers; persistent transient → `stt_failed`/`mt_failed` segment, continue. Never touches the capture thread. |
| 401/403 or quota 402 | Log once; stop new API calls but keep capturing/sealing to disk. Ctrl+C message points at recovery (§ below). |
| `--max-cost INR` reached (default 50) | Same: stop API calls, keep capturing, log once. Accrued cost tallied from actual API calls. |
| Capture child dies mid-session | `CaptureError` → seal buffer (if ≥ min_s), append what workers finished, valid file, exit 2. |
| Ctrl+C | First: seal buffer (≥ min_s) → transcribe → append → final header → exit 0 (2 if any segment not `ok`). Second Ctrl+C: immediate exit; file remains valid by construction. SIGINT handler reaps the ffmpeg child explicitly (both are in the terminal's process group). |

**Quota recovery**: the batch command gains `omnilingual transcribe --from-chunks <session-dir>`: reads the live session's `chunks.json` + cache and finishes remaining chunks with zero re-payment (batch's `work_dir_for` hashes a source file, so the live cache is otherwise unreachable — this flag is the bridge, and `session.json` validates the dir).

## 8. Output format and crash-safe writing

Identical segment body to v1 §8. File layout: a fixed-width, space-padded header block (4 lines, constant byte width) updated **in place** via `pwrite` after each append — stable inode, so `tail -f` never loses the file (tmp+rename would swap the inode). Segment bodies are single `O_APPEND` writes, `fsync` after each. After a kill the header may undercount bodies (safe direction); a live session never truncates or rewrites an existing file — if `--out` already exists, the run writes to `<stem>-<HHMM>.md` next to it (e.g. `meeting-1530.md`), keeping every prior transcript intact. The work dir is always fresh (timestamped).

```markdown
# Meeting transcript — LIVE 2026-09-05 15:30 IST
Duration (so far) 00:14:20 · 31 segments · hi-IN 58% en-IN 42%
Cost ₹14.60 (cap ₹50) · growing

## Transcript
**[00:00:04 → 00:00:13] hi-IN**
हम आज payment dashboard के बारे में बात करेंगे...
> We will talk about the payment dashboard today...
```

On clean stop the header becomes final (cap line loses `· growing`). `--english-only` maintains `<stem>.en.md` incrementally with the same scheme.

## 9. CLI

```
omnilingual live --out meeting.md
    --input "Omnilingual"        # aggregate device, substring match (default "Omnilingual")
    --mic-only                   # degraded single-device mode (no BlackHole)
    --langs hi-IN,ta-IN,en-IN
    --target-s 8                 # min_s <= target_s <= max_s
    --max-chunk-s 28 --min-chunk-s 5 --noise-db -35
    --stt-workers 2
    --max-cost 50                # INR; stop API calls beyond this
    --check-audio                # devices + 1 s RMS probe; no API calls
    --api-key KEY --work-dir PATH -v
```

Status lines to stderr only: `[live 00:03:12] sealed #14 (9.2 s) → hi-IN 0.97 → en ✓`, lag every 30 s, cost cap warnings.

Disk: sealed WAVs accumulate (~7 MB/min); session dirs are not auto-cleaned (same policy as v1).

## 10. Testing

- **Unit**: `live_slicer` on synthetic PCM with known gaps → seals on FIRST gap after target, none < min or > max, full coverage; byte-clock math; partial flush on close; energy-gate skip path. `live_capture` device-list parsing (incl. ambiguous-name error) against recorded `-list_devices` output. `render/live` kill-mid-append → file parses, bodies intact, header ≤ bodies; inode stability across updates.
- **Integration (respx)**: 60 s synthetic clip through `run_live` → golden Markdown body-equal to batch golden; repeated bytes → zero new HTTP calls; slow-STT mock → ordered appends despite out-of-order completion + growing lag log.
- **Failure**: capture child `kill -9` mid-session → partial seal, valid file, exit 2.
- **Manual (@pytest.mark.live / documented)**: mic loopback 30 s; one real STT+MT call; TCC prompt path via `tccutil reset Microphone`; system-audio path = known clip through BlackHole diffed against its file transcription; 48 kHz vs 16 kHz mismatch check.
- TDD per module, mirroring v1 §10.

## 11. Setup UX (macOS, one time)

1. `brew install blackhole-2ch`.
2. Audio MIDI Setup ▸ `+` ▸ **Multi-Output Device**: headphones + BlackHole 2ch. Set system output to it. Note: volume keys won't work — set volume first.
3. Audio MIDI Setup ▸ `+` ▸ **Aggregate Device**: mic + BlackHole 2ch; clock source BlackHole; drift correction on mic. Rename to `Omnilingual`.
4. Wear headphones (speakers → echo double-transcription).
5. `omnilingual live --check-audio`, then `omnilingual live --out standup.md`.

## 12. Non-goals (explicitly out)

Speaker tags/diarization, per-remote-participant labels, sub-5 s subtitles, TUI, Linux/Windows, Notion export, summaries. The `LiveCapture` protocol and unchanged `Segment` model keep those doors open.
