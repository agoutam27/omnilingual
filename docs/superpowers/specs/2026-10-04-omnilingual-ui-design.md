# Omnilingual UI — design spec

**Date:** 2026-10-04
**Status:** approved in brainstorming; ready for implementation planning
**Scope:** a local desktop-style UI over the existing `omnilingual` package, in two shippable phases

---

## 1. What exists today

`omnilingual` is a Python 3.12+ CLI (uv-managed, macOS) that transcribes and translates
speech. Two commands:

- `omnilingual transcribe RECORDING` — batch. Normalizes with ffmpeg, splits into chunks,
  runs STT then MT per chunk, renders Markdown.
- `omnilingual live` — captures from a macOS Aggregate Device via ffmpeg, slices on
  silence, transcribes and translates concurrently, and appends to a Markdown file that
  stays valid at every instant.

Both already accept the full parameter surface (`--stt`, `--stt-model`, `--mt`,
`--mt-model`, `--diarize`, `--diarizer`, `--speakers`, `--langs`, `--work-dir`,
`--max-chunk-s`, `--min-chunk-s`, `--api-key`, `--english-only`, plus `--check-audio`,
`--mic-only`, `--target-s`, `--noise-db`, `--stt-workers`, `--max-cost`, `--from-chunks`
for live). `--ask` is an interactive wizard over that same surface.

`scripts/setup-mac.sh` installs and configures everything (Xcode CLT, Homebrew, ffmpeg,
BlackHole, the `Omnilingual` Aggregate Device, optional model prefetch, API keys into a
gitignored `.env`), prompts for capabilities, persists its answers, and converges on
re-run.

### The integration surface (why no subprocess is needed)

The package is importable and already callback-driven. A UI never has to parse terminal
output:

| Need | Existing API |
|---|---|
| Batch streaming | `run(source, work_dir, settings, stt, translator, cache, progress: Progress \| None, diarizer=None) -> Transcript` — `Progress = Callable[[int, int, Segment], None]` (`pipeline/__init__.py:28,298`) |
| Live progress text | `run_live(..., status: Callable[[str], None] \| None = None)` — **pre-rendered strings with no transcript text** |
| Live transcript rows | **none** — text reaches only the Markdown file via a `writers` list local to `run_live` |
| Live stop | **none** — Ctrl+C only, via `signal.signal(SIGINT, …)` at `pipeline/live.py:383` |
| Live recovery | `run_from_chunks(session_dir, settings, stt, translator, cache, progress, diarizer)` |
| Providers | `load_settings(api_key=None, env=None, **overrides) -> Settings`, `build_stt`, `build_translator`, `build_diarizer` |
| Output rendering | `render.markdown.render(t) -> str`, `render.markdown.render_english_only(t) -> str` — pure, no I/O |
| Row model | `Segment(chunk, lang, prob, text, english, status, speaker)` — exactly one UI row |
| Audio devices | `scripts/audio-devices.swift` (`list`, `check`, `ensure`) |
| Test seam | `run_live(..., capture_factory=LiveCapture)` already injectable |

Two gaps must be closed, both small and both in one file (§4).

**Blocking constraint, verified empirically:** `signal.signal()` raises
`ValueError: signal only works in main thread of the main interpreter` when called off
the main thread. `run_live` calls it unconditionally, so running it in a worker thread —
the natural thing for a server — fails today. `signal.getsignal()` is safe from any
thread; only the *registration* needs guarding.

---

## 2. Goals

1. One window that runs a live session or transcribes a recording, with every parameter
   selectable before the run starts.
2. Live mode shows the translation appearing as it is produced. Recording mode shows the
   finished translated transcript.
3. Audio capture readiness is checked automatically, with a one-click fix when it is not.
4. Installation is reachable from the UI, by driving the existing `setup-mac.sh` — not by
   reimplementing it.
5. One run at a time per window. Stop is always available and always leaves valid output.

## 3. Non-goals

- No multi-user, no remote access, no cloud deployment. Loopback only.
- No editing of transcripts in the app; the Markdown file is the artefact and the UI
  opens it in the default editor.
- No re-implementation of setup, diarization, chunking, or rendering. All of it is reused.
- No attempt to be a general audio recorder. Live mode is a capture surface for
  transcription, exactly as the CLI is.
- Phase 1 does not touch the CLI's behaviour or flags.

---

## 4. The one change to the existing package

`pipeline/live.py`, `run_live` only. Signature gains two keyword-only params:

```python
def run_live(opts: LiveOptions, settings, stt, translator, *,
             diarizer=None,
             status: Callable[[str], None] | None = None,
             capture_factory: Callable = LiveCapture,
             on_segment: Callable[[Segment, bool], None] | None = None,
             stop_event: threading.Event | None = None) -> int:
```

**(a) Adopt an externally-owned stop flag.** Line 210 changes from
`stop = threading.Event()` to `stop = stop_event or threading.Event()`. `capture_loop`
already tests `stop.is_set()` (line 242), so it honours the flag with no further change,
and the existing graceful-shutdown path is untouched. Verified: an externally-created
`Event` behaves identically to an internal one.

**(b) Guard SIGINT registration so the host owns signals.** Lines 369–383 and the restore
at 450 become:

```python
    # Two-stage Ctrl+C: first stops gracefully, second reaps ffmpeg and exits.
    # Only a main-thread host can install a handler; a GUI server owns SIGINT itself
    # and stops the run through stop_event instead.
    sigints = 0
    prev = signal.getsignal(signal.SIGINT)
    owns_sigint = threading.current_thread() is threading.main_thread()

    def on_sigint(signum, frame):
        nonlocal sigints
        sigints += 1
        if sigints == 1:
            say(f"[live {elapsed()}] stopping… (Ctrl+C again to quit now)")
            stop.set()
        else:
            capture.close()
            os._exit(2)

    if owns_sigint:
        signal.signal(signal.SIGINT, on_sigint)
```

and in `finally`, `if owns_sigint: signal.signal(signal.SIGINT, prev)`.

Both the registration and the restore are guarded: an unguarded restore would raise in
the same thread whose registration was skipped, masking the real exit path.

**(c) Emit structured rows.** Next to the existing `say = status or (lambda msg: None)`,
add `notify = on_segment or (lambda seg, keep: None)`. In the ordered emission loop,
after `keep` is computed (lines 426–434):

```python
            seg, delta, _ = item
            say(status_line(state["next"] - 1, seg))
            keep = True
            if (seg.status == "no_speech"
                    and seg.chunk.end_s - state["last_kept_end"] <= 30.0):
                keep = False
            if on_segment is not None:
                try:
                    on_segment(seg, keep)
                except Exception:  # a UI callback must never end the session
                    log.exception("on_segment callback failed")
            if keep:
                ...
```

Two decisions encoded here:

- **The callback fires for dropped silence segments too**, carrying `keep=False`. The
  status line already reports every sealed chunk (`→ silence`), and a live view that
  freezes during a quiet stretch looks broken. The UI renders `keep=False` rows dimmed
  and out of the saved file, which is exactly what the file does.
- **Callback exceptions are swallowed and logged.** This callback crosses a thread
  boundary into UI code that the session cannot vouch for; a rendering bug must not kill
  a meeting that is being transcribed.

`status=` keeps its current signature and behaviour. Batch mode is untouched: `run`
already streams `Segment`s through `progress`.

**Blast radius:** `run_live` has 12 call sites, all in `cli.py`, and two test modules
(`tests/pipeline/test_run_live.py`, `tests/pipeline/test_recovery.py`). Both new params
are keyword-only with `None` defaults, so every existing caller is source-compatible.

---

## 5. New package layout

`omnilingual/ui/` — new, and the only other thing added in phase 1.

| File | Responsibility | Depends on |
|---|---|---|
| `__main__.py` | `main()`: parse `--host/--port/--no-window/--open-browser`, pick a free port, start uvicorn in a daemon thread, open a pywebview window (fall back to `open <url>`), block until the window closes, then shut down cleanly | `server`, `pywebview` |
| `server.py` | The ASGI app: REST routes, the WebSocket endpoint, request validation, static files. Owns no transcription logic | `session`, `settings`, `audio`, `secrets` |
| `session.py` | `SessionRunner` — one per run. Owns the worker thread, the asyncio queue, the bridge from pipeline callbacks into the queue, and the run lifecycle state machine | `pipeline`, `config`, `render`, `stt`/`translate`/`diarize` factories |
| `settings.py` | Read/write `~/.config/omnilingual/ui.toml`; typed accessors for every run parameter | — |
| `secrets.py` | Which API keys are present (booleans only) and writing/clearing a key in the repo `.env` | — |
| `audio.py` | Device readiness probe and the fix action, wrapping `scripts/audio-devices.swift` | `subprocess`, `swiftc` |
| `static/index.html`, `app.js`, `style.css` | The single page. `app.js` is a thin renderer over WebSocket messages; no business logic, no transcription logic | — |

Boundaries that matter: `session.py` is the only module that knows how a run executes, so
swapping the in-process thread (§6) for a child process later is a single-class change.
`server.py` never touches `pipeline`. Nothing in `ui/` is imported by the CLI, so the CLI
cannot acquire a web dependency by accident.

---

## 6. How a run executes

### Chosen: in-process worker thread + WebSocket

The session runs in a thread inside the server process. `on_segment` and `status` push
into an `asyncio.Queue` via `loop.call_soon_threadsafe`; a WebSocket drains it to the
page.

Chosen because `run_live`'s own threads are already daemon threads, the capture path is
I/O-bound, and the callbacks plug straight in with no second process boundary and no
event-emitting CLI.

Rejected: **child process per run with JSONL events on stdout.** It survives a wedged
transcription and outlives a page reload, at the cost of a process boundary, a runner
that must emit events, and duplicated setup. Worth revisiting only if the UI ever needs
to survive a hard hang; the `SessionRunner` boundary is chosen so that swap is local.

**Known limitation, accepted:** a genuine hang inside the pipeline cannot be interrupted
from the UI and requires force-quitting the app. `run_live`'s threads are daemons and the
`cond.wait(timeout=0.5)` loop is bounded, so the realistic failure modes (a wedged ffmpeg,
an HTTP call with no timeout) are already bounded today.

---

## 7. HTTP + WebSocket contract

All routes are `127.0.0.1`-only.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | `{ok, version, arch, macos, arm64, python}` |
| GET | `/api/defaults` | Saved `ui.toml` + `STT_PROVIDERS`, `MT_PROVIDERS`, `DEFAULT_STT_MODELS`, `DEFAULT_MT_MODELS`, the `LiveOptions` defaults, and the last-used output directory |
| PUT | `/api/settings` | Persist `ui.toml` (validated; unknown keys rejected) |
| GET | `/api/keys` | `{"SARVAM_API_KEY": true, "GROQ_API_KEY": true, "GEMINI_API_KEY": false}` — **booleans only; values never leave the process** |
| PUT | `/api/keys` | `{"GROQ_API_KEY": "sk-…"}` or `{"GROQ_API_KEY": null}` to clear — writes through `secrets.py` |
| GET | `/api/audio` | Readiness (below) |
| POST | `/api/audio/setup` | Build the helper, wait for BlackHole, `ensure`, `check`; progress on the WebSocket log channel |
| POST | `/api/audio/restart-daemon` | `killall coreaudiod` via `osascript … with administrator privileges` (the macOS auth prompt) |
| POST | `/api/session/start` | Validate, build providers, start a `SessionRunner`; `200 {run_id, mode}` or `400 {error}` |
| POST | `/api/session/stop` | Set the run's `stop_event` |
| GET | `/api/runs` | Recent runs from `.omnilingual/live-*` session dirs and the outputs dir |
| POST | `/api/session/recover` | `run_from_chunks` on a past session dir |
| GET | `/api/setup/preview` | Runs `scripts/setup-mac.sh --dry-run --yes`, returns its output text |
| POST | `/api/setup/apply` | Runs `scripts/setup-mac.sh --yes`, streaming output on the WebSocket log channel |
| WS | `/ws` | The event stream (§7.1) |

### 7.1 WebSocket messages

Server → client:

```json
{"type":"state","run_id":"…","mode":"live","status":"running",
 "elapsed_s":812.4,"cost_inr":3.18,"cost_cap":50.0,
 "segments":41,"dropped":6,"out":"/abs/standup.md","session_dir":"/abs/.omnilingual/live-2026…"}
{"type":"segment","seq":41,"idx":40,"start_s":800.1,"end_s":808.3,
 "lang":"hi","prob":0.91,"text":"…","english":"…","status":"ok",
 "speaker":"S1","kept":true,"cost_inr":0.08}
{"type":"status","message":"[live 00:13:32] sealed #40 (8.2 s) → hi 0.91 → en ✓"}
{"type":"log","level":"info|warn|error","message":"…"}
{"type":"end","exit_code":0,"out":"…","session_dir":"…","recoverable":false}
```

`status` is one of `idle | running | stopping | done | failed | halted`. `halted` means
capture continued but API calls stopped (cost cap, quota, auth) and the session is
recoverable — the `end` message carries `recoverable: true` and the UI offers recovery.

Client → server: `{"type":"stop"}` and `{"type":"recover","run_id":"…"}`.

`segment.kept` is `true` for every batch segment (batch keeps all) and mirrors live's
`keep` decision for live. `english` is `null` for `en-IN` source, for failures, and for
silence — the UI shows the original text alone in those cases, as `format_segment` does.

### 7.2 Local-server hardening

The server binds `127.0.0.1` only, but any page in the user's browser can attempt a
request to a loopback port. Three cheap defences, all required:

1. A random token minted per launch, embedded in `index.html` and required in an
   `X-Omnilingual-Token` header on every `/api/*` and the WebSocket. Absent or wrong →
   `403`.
2. Reject requests whose `Host` header is not `127.0.0.1:<port>` or `localhost:<port>`
   (blocks DNS-rebinding).
3. Reject any request carrying a foreign `Origin` header.

No CORS middleware is installed, so browsers block cross-origin reads by default; the
token covers the writes.

---

## 8. Run lifecycle

### Live

1. `POST /api/session/start` validates the request (§9), builds
   `Settings`/`STTProvider`/`Translator`/`Diarizer` **before** returning, so a missing key
   or missing extra is a `400`, not a mid-session failure.
2. A `SessionRunner` starts a daemon thread running `run_live(opts, settings, stt,
   translator, diarizer=…, status=self._on_status, on_segment=self._on_segment,
   stop_event=self._stop)`.
3. `on_segment` → `loop.call_soon_threadsafe(queue.put_nowait, …)` → the WebSocket drains
   it → the page appends a row.
4. The run writes `standup.md` continuously, exactly as the CLI does. The file is valid at
   every instant, including after a stop.
5. Stop sets `_stop`; the same graceful path as Ctrl+C-first-press runs; `run_live`
   returns `0`/`1`/`2`; the writer finalises; `end` is sent.
6. On start the UI reports `session_dir`, so recovery is available even after a quit.

### Recording

1. Same validation and provider construction.
2. A thread runs `run(source, work_dir, settings, stt, translator, cache,
   progress=self._on_progress, diarizer=…)`, where `_on_progress(i, total, seg)` becomes
   a `segment` message with `kept: true`.
3. The returned `Transcript` is rendered with `render.markdown.render` (and
   `render_english_only` when `--english-only`) and written to `--out`. **The UI renders
   nothing itself** — the file is byte-identical to what the CLI produces, and the page
   displays that same text.
4. Progress is `i/total`, so the page shows a determinate bar.

### Recovery

`POST /api/session/recover` with a `session_dir` runs `run_from_chunks(…,
progress=self._on_progress, …)` and renders to a new output file. Already-billed chunks
cost nothing because the session's own `cache/` is the namespace — this is existing
behaviour, surfaced as a button.

### Concurrency

One run at a time. `POST /api/session/start` while a run is active → `409`. The
`SessionRunner` registry lives on the app state, keyed by `run_id`.

---

## 9. Parameters and validation

The panel is exactly the CLI's flag surface — no invented options:

| Panel control | Field | Default |
|---|---|---|
| Mode | `mode` | `live` |
| Recording (recording mode only) | `source` | — |
| Output file | `out` | `standup.md` |
| Speech-to-text | `stt`, `stt_model` | `sarvam`, provider default |
| Translate | `mt`, `mt_model` | `mayura`, provider default |
| Speaker labels | `diarize`, `num_speakers` | off, `3` |
| Languages (blank = auto-detect) | `langs` | `[]` |
| English-only view | `english_only` | off |
| Work directory | `work_dir` | `.omnilingual` |
| Device (live) | `device` | `Omnilingual` |
| Microphone only (live) | `mic_only` | off |
| Target chunk seconds (live) | `target_s` | `8.0` |
| Chunk bounds (live) | `max_chunk_s`, `min_chunk_s` | `28.0`, `5.0` |
| Noise floor dBFS (live) | `noise_db` | `-35.0` |
| STT workers (live) | `stt_workers` | `2` |
| Cost cap ₹ (live) | `max_cost` | `50.0` |

**The panel cannot produce a configuration the CLI would reject.** Validation is not
reimplemented. The server calls `load_settings(..., **overrides)`, which already raises
`ConfigError` for an unknown STT/MT provider, for `num_speakers < 2`, and eagerly for
both provider models. The chunk-bounds rule (`max_chunk_s < MAX_CHUNK_LIMIT_S` and
`max_chunk_s >= 2 × min_chunk_s`, with `MAX_CHUNK_LIMIT_S = 30.0` currently declared in
`cli.py:63`) is **extracted from `cli.py` into a shared `validate_chunk_bounds` helper**,
and both the CLI and the UI call it. It is not copied into `ui/server.py`, because two
copies of a numeric rule drift. This is the only refactor of existing code in phase 1, and
it is a pure extraction — no behaviour change.

`--api-key` is deliberately absent from the panel: keys live in `.env`, and the panel
edits them there.

A live-specific readiness pre-check runs before `start` in live mode and requires all four
of: the `Omnilingual` device present, BlackHole present, `ffmpeg` and `ffprobe` on `PATH`,
and `mic_authorized` true. Failing any returns `400` with the specific remedy, and the UI
shows the corresponding banner + fix button rather than starting a session that cannot
work. The one-second microphone probe is the only costly item here, which is why it runs
once at startup (and on demand after a fix) rather than on every render.

---

## 10. Settings and secrets

**Run preferences:** `~/.config/omnilingual/ui.toml` — directory `0700`, file `0600`,
written atomically via a `mkstemp` sibling plus `os.replace`. Contents mirror the table in
§9. Loaded on startup, saved on change, and used as the panel's defaults so the UI opens
where the user last worked. The CLI does **not** read this file in phase 1 (that is a
follow-up, listed in §16); it is the UI's own store, in the same directory and with the
same permissions as the existing `setup-mac.conf`.

**API keys stay in the repo's gitignored `.env`**, per the existing convention. The UI
never displays a stored key value; `GET /api/keys` returns booleans, and the panel shows
"configured / not configured" with a field to type a new value or clear it.

`ui/secrets.py` reimplements, in Python, the same `.env` semantics `setup-mac.sh` already
provides and that were verified during the script's own review: preserve comments and
unrelated keys, replace an existing key **in place** rather than appending a duplicate,
keep mode `0600`, write atomically, and never pass a value to a subprocess (so it cannot
appear in `ps`) or echo it. This is a deliberate ~20-line cross-language duplication —
the bash implementation cannot be imported, and the alternative (shelling out to the
script for one line of work) would be worse.

---

## 11. Audio readiness

`GET /api/audio` returns:

```json
{"ok": true,
 "device": "Omnilingual",
 "blackhole": true,
 "ffmpeg": true,
 "ffprobe": true,
 "mic_authorized": true,
 "output": "Multi-Output Device",
 "detail": []}
```

- `device` — `"Omnilingual" list` from the compiled `audio-devices.swift` helper.
- `blackhole` — BlackHole 2ch present in the device list.
- `ffmpeg` / `ffprobe` — both on `PATH`. Both are genuinely required: `ensure_ffmpeg()`
  is called from both the batch and live paths.
- `mic_authorized` — whether the *Python process* holds macOS Microphone TCC permission.
  There is no supported way to read TCC status directly, so it is determined
  **empirically**: the probe runs a one-second silent capture,
  `ffmpeg -f avfoundation -i "<device>" -t 1 -f null -`, and treats a non-zero exit, or
  `Permission denied` / `not permitted` / `AVFoundation: ... error` in stderr, as denied.
  This is sound because capture runs through ffmpeg in this process rather than the
  browser's `getUserMedia`, so there is no browser permission prompt — but macOS still
  requires that the launching terminal or app be approved in System Settings → Privacy &
  Security → Microphone, and it refuses ffmpeg's capture at the AVFoundation layer, which
  is exactly what the probe observes. A denied probe is reported as `false` with the
  System Settings pane named, instead of surfacing later as an opaque ffmpeg failure.
- `output` — current system output, via `SwitchAudioSource -c`, shown only so the user can
  see why meeting audio is not reaching BlackHole.

**Startup behaviour:** probe once. If `ok` is false, show a dismissible banner naming what
is missing. **Never switch system output automatically** — the Multi-Output Device route
silently breaks volume keys, which is the maintainer's own documented caveat. The banner
offers two explicit buttons:

- **Set up audio** → `POST /api/audio/setup`: compile the helper, wait for BlackHole,
  `ensure`, `check`. Same steps as `setup-mac.sh`, idempotent.
- **Restart audio daemon** → only offered when the device is missing on a fresh install,
  because BlackHole needs `coreaudiod` restarted to appear. The UI has no TTY, so
  `setup-mac.sh` takes its documented no-sudo branch; this button performs just that one
  step through `osascript -e 'do shell script "killall coreaudiod" with administrator
  privileges'`, which raises the standard macOS authorisation dialog.

---

## 12. Phase 2 — the Setup screen

Phase 2 does not reimplement installation. It drives `scripts/setup-mac.sh`, which is
already idempotent, prompting, re-runnable, arch-aware, and reviewed.

- **Preview** — `GET /api/setup/preview` runs `setup-mac.sh --dry-run --yes` and shows the
  plan. Nothing is modified.
- **Apply** — `POST /api/setup/apply` runs `setup-mac.sh --yes`, streaming stdout+stderr
  line by line into the same log strip used for run output. Because the script runs with
  no TTY, its `sudo` branch is skipped and it prints its own reboot hint; the UI shows
  that hint as a note rather than hiding it.
- Answers to the script's capability prompts come from the UI's own form, passed as
  explicit flags. The script's interactive prompts are not driven programmatically —
  keeping the script's prompt surface intact is worth more than avoiding three flags.

**Required follow-up edit to `setup-mac.sh`:** adding a `ui` extra to `pyproject.toml`
means the script's `ALL_EXTRAS` constant must gain `ui`, otherwise `--all-extras` installs
it unconditionally and it can never be deselected on re-run — exactly the convergence
property the script was built for. The capabilities prompt gains a matching entry.

**Restart after apply.** The UI is running from the `.venv` that `uv sync` just mutated,
and its already-imported modules are stale: a newly installed extra will not be importable
in-process. After a successful apply the UI therefore shows "Restart to use newly
installed components" with a relaunch button, rather than pretending the change is live.

---

## 13. Error handling

Every failure mode lands in the log strip with the exception's own message, never a dialog
and never a stack trace in the page.

| Condition | Handling |
|---|---|
| `ConfigError` from `load_settings` | `400` at start with the message verbatim; the offending field is highlighted in the panel |
| `FfmpegMissingError` | Start is refused; banner names `brew install ffmpeg` |
| `CaptureError` (device missing) | Start is refused in live mode; banner + **Set up audio** |
| Microphone TCC denied | Banner + link to the System Settings pane |
| Missing extra (`mlx_whisper`, `sherpa_onnx`, `ctranslate2`) | Detected at start; renders as "Run Setup to add speaker diarization" — a button, not a traceback |
| `QuotaError` / `AuthError` / cost cap | `status: "halted"`, capture continues, `end.recoverable: true`, recovery button offered |
| Pipeline exception in the session thread | `status: "failed"`, the message is sent, the run registry is cleaned up, and the app stays usable |
| Exception inside an `on_segment` callback | Logged and swallowed (§4c) — the session continues |
| WebSocket disconnect mid-run | The run continues to completion and the file stays valid; on reconnect the UI fetches final state via `/api/runs` |

The last row is why the run lives in the server rather than the page: closing the window
does not destroy a transcription in progress.

---

## 14. Testing

The seams already exist, so none of this needs hardware.

**`pipeline/live.py` (direct tests, `tests/pipeline/`):**
- `on_segment` fires once per emitted segment, in order, with the correct `keep` flag —
  including a dropped `no_speech` segment inside the 30 s window.
- `on_segment` raising does not end the run.
- `stop_event` supplied by the caller stops the run (scripted `capture_factory`).
- `run_live` still works with both params omitted, and from a non-main thread without
  `ValueError` — the regression test for the SIGINT guard.
- The CLI's existing two-stage Ctrl+C behaviour is unchanged when run on the main thread.

**`ui/` (new, `tests/ui/`):**
- `SessionRunner` against a scripted `capture_factory` and stub STT/MT: asserts the exact
  sequence of WebSocket messages for a known chunk list. No audio hardware, no network.
- Server routes via an ASGI test client: defaults round-trip, `PUT /api/settings` rejects
  unknown keys, `/api/session/start` returns `400` for a bad provider and for a missing
  key, `409` while a run is active.
- The token/Host/Origin defences: a request without the token is `403`, a foreign `Host`
  is rejected.
- `secrets.py` against a fixture `.env`: comments and unrelated keys preserved, an existing
  key replaced in place with no duplicate, mode stays `0600`, a cleared key removed.

**Not tested automatically:** the pywebview window itself, and the System Settings
microphone flow. Both are manual, and both have a documented fallback (browser instead of
window; banner plus link).

`uv run pytest -q` must stay green; the opt-in real-model gates (`--run-live`,
`--run-mlx`, `--run-faster-whisper`, `--run-diarize`) are unaffected because the new
params default to `None`.

---

## 15. Dependencies and packaging

New optional extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
ui = ["fastapi>=0.115", "uvicorn[standard]>=0.32", "pywebview>=5.0"]

[project.scripts]
omnilingual-ui = "omnilingual.ui.__main__:main"
```

`uvicorn[standard]` brings the WebSocket implementation, so no separate `websockets`
pin. Core deps are untouched, so the CLI does not gain a web dependency.

`pywebview` is the only platform-specific piece and is macOS-only by intent; if its import
fails, `__main__.py` falls back to `open <url>` and the app is fully usable in a browser.
The server binds `127.0.0.1` on an ephemeral port chosen at launch, so multiple instances
never collide and nothing is exposed to the network.

---

## 16. Phasing, acceptance, and risks

This is one spec covering two passes, planned and delivered separately: **implementation
plan #1 covers phase 1 only**, and plan #2 covers phase 2. Phase 2 depends on phase 1's
window, WebSocket log channel, and settings store existing, so the order is fixed; nothing
else about phase 1 constrains it. Each phase is independently shippable and leaves the CLI
working exactly as it does today.

### Phase 1 — runtime

Ship when:

1. `uv run omnilingual-ui` opens a window (or browser) and `GET /api/health` is `ok`.
2. A live session against the `Omnilingual` device streams rows to the page while
   `standup.md` grows alongside, and the two agree on segment count.
3. Stop leaves a valid `standup.md` and a `done`/`end` message.
4. A recording renders to a file byte-identical to the CLI's, given the same parameters.
5. Halting on cost cap or quota yields `recoverable: true` and recovery works.
6. Audio banner appears when the device is absent, and **Set up audio** fixes it.
7. Every row in §14 passes.

### Phase 2 — setup screen

Ship when:

1. Preview shows the script's plan and changes nothing on disk.
2. Apply streams output and completes.
3. After applying an extras change, the UI says a restart is needed and relaunches cleanly.
4. Re-running apply after changing the panel's capabilities converges, with no duplicate
   installs — the property already proven for the script on its own.

### Risks

| Risk | Mitigation |
|---|---|
| `run_live`'s unconditional `signal.signal` breaks the server | Fixed in phase 1 by design (§4b), with a regression test |
| A pipeline hang is uninterruptible from the UI | Accepted (§6); the `SessionRunner` boundary makes a child-process runner a local change later |
| Adding the `ui` extra silently breaks `setup-mac.sh` convergence | Called out in §12 as a required edit, not an optional one |
| pywebview unavailable or broken on a future macOS | Browser fallback; the app never depends on the window |
| Duplicated `.env` logic in Python and bash drifting | Both are small and independently tested; the invariants are restated in each module's docstring |

### Deliberately out of scope for now

- Having the CLI read `ui.toml` for defaults, so terminal and window always agree. Worth
  doing; it is a behaviour change to the CLI and belongs in its own change.
- Multi-window or concurrent sessions.
- Editing or re-running a transcript from the UI.