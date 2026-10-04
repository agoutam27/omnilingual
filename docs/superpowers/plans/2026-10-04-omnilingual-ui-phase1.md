# Omnilingual UI Phase 1 (runtime) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a local desktop-style UI that runs a live session or transcribes a recording, with every parameter selectable before the run and audio readiness checked (with a one-click fix) up front.

**Architecture:** The `omnilingual` package is already importable and callback-driven, so the UI runs a session **in-process in a worker thread** and streams structured events to one HTML page over a WebSocket — no subprocess, no terminal-output parsing. Exactly one existing file changes (`pipeline/live.py`: two keyword-only params plus a `signal.signal` guard) so `run_live` can be driven from a non-main thread and emit structured rows. A new `omnilingual/ui/` package owns everything else; the recording path reuses `render.markdown.render()` so its output file is byte-identical to the CLI's.

**Tech Stack:** Python 3.12+, `uv`, FastAPI + `uvicorn[standard]` (ASGI + WebSocket), `pywebview` (window, with browser fallback), `pytest` + `respx`, and the existing `omnilingual` pipeline/config/render/stt/translate/diarize factories.

**Spec:** `docs/superpowers/specs/2026-10-04-omnilingual-ui-design.md` (approved, commit `c5957b2`).

## Global Constraints

These are requirements from the spec. Every task implicitly includes them.

- The server binds `127.0.0.1` only, on an ephemeral port chosen at launch. Never expose to the network.
- Loopback hardening is mandatory, all three: a per-launch random token on every `/api/*` route and the WebSocket (absent/wrong → `403`); reject any `Host` header that is not `127.0.0.1:<port>` or `localhost:<port>`; reject any foreign `Origin`. No CORS middleware.
- API key values **never** leave the process. `GET /api/keys` returns booleans only. Keys live in the repo's gitignored `.env` — never in the UI settings file, never in an argv, never logged or echoed.
- Never switch the system audio output automatically: the Multi-Output Device route silently breaks the volume keys.
- The UI renders no transcript formatting of its own. Recording output goes through `render.markdown.render()` / `render_english_only()`; the page displays that same text.
- The parameter panel is exactly the CLI's flag surface. No invented options. Validation is never reimplemented — it calls `load_settings()` and the shared chunk-bound validators.
- One run at a time per window. `POST /api/session/start` while a run is active → `409`.
- A WebSocket disconnect must NOT kill a running session. The run lives in the server, not the page.
- `session.py` is the ONLY module that knows how a run executes, so a future child-process runner is a single-class change. `server.py` never imports `omnilingual.pipeline`.
- Nothing in `omnilingual/ui/` may be imported by the CLI, so the CLI cannot acquire a web dependency.
- All run threads are daemon threads. A run always leaves a valid output file, including after a stop.
- `uv run pytest -q` must stay green throughout. The opt-in real-model gates (`--run-live`, `--run-mlx`, `--run-faster-whisper`, `--run-diarize`) must be unaffected.
- Phase 1 must not change CLI behaviour or flags. The only refactor of existing code is the pure extraction in Task 1.
- Repo root, from inside the package, is `Path(__file__).resolve().parents[2]`.

---

### Task 1: Extract the chunk-bounds rule into shared validators

The CLI enforces a numeric chunk-bound rule in two places. The UI must enforce the *same* rule, and two copies of a numeric bound drift. Extract it into `config.py`, where `ConfigError` already lives, so the CLI's `_fail` path and the UI's `400` path see the same exception type.

**Files:**
- Modify: `omnilingual/config.py` (add `MAX_CHUNK_LIMIT_S`, `validate_chunk_bounds`, `validate_target_s`)
- Modify: `omnilingual/cli.py` (delete the constant and its comment at lines 62-63; replace the check at ~336-337 and the two checks at ~428-431)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (existing module).
- Produces:
  - `config.MAX_CHUNK_LIMIT_S: float` — `30.0`, moved out of `cli.py`.
  - `config.validate_chunk_bounds(min_chunk_s: float, max_chunk_s: float) -> None` — raises `ConfigError`.
  - `config.validate_target_s(target_s: float, min_chunk_s: float, max_chunk_s: float) -> None` — raises `ConfigError`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_validate_chunk_bounds_accepts_cli_defaults():
    validate_chunk_bounds(5.0, 28.0)


@pytest.mark.parametrize("mins,maxs", [
    (0.0, 28.0),      # min not positive
    (-1.0, 28.0),     # min negative
    (5.0, 4.0),       # max below min
    (5.0, 5.0),       # max == min
    (5.0, 30.0),      # max at the Sarvam ceiling
    (5.0, 40.0),      # max above the ceiling
    (5.0, 9.0),       # max < 2x min
])
def test_validate_chunk_bounds_rejects(mins, maxs):
    with pytest.raises(ConfigError):
        validate_chunk_bounds(mins, maxs)


def test_validate_chunk_bounds_message_keeps_cli_wording():
    # tests/test_cli_live.py asserts on this substring, so the wording is load-bearing.
    with pytest.raises(ConfigError) as exc:
        validate_chunk_bounds(5.0, 40.0)
    assert "--max-chunk-s must be < 30" in str(exc.value)


def test_validate_target_s_accepts_value_inside_bounds():
    validate_target_s(8.0, 5.0, 28.0)
    validate_target_s(5.0, 5.0, 28.0)
    validate_target_s(28.0, 5.0, 28.0)


@pytest.mark.parametrize("target", [4.9, 28.1])
def test_validate_target_s_rejects_outside_bounds(target):
    with pytest.raises(ConfigError):
        validate_target_s(target, 5.0, 28.0)
```

Replace the import at the top of `tests/test_config.py` with:

```python
from omnilingual.config import (
    ConfigError,
    Settings,
    load_settings,
    validate_chunk_bounds,
    validate_target_s,
)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -q -k validate`
Expected: FAIL — `ImportError: cannot import name 'validate_chunk_bounds'`.

- [ ] **Step 3: Add the validators to `config.py`**

Append to `omnilingual/config.py`:

```python
# Sarvam's synchronous speech-to-text endpoint rejects audio of 30 seconds or
# more, which is what makes a ceiling necessary rather than arbitrary.
MAX_CHUNK_LIMIT_S = 30.0


def validate_chunk_bounds(min_chunk_s: float, max_chunk_s: float) -> None:
    """Raise ConfigError unless these bounds can produce valid chunks.

    Lives in config rather than cli so the CLI and the UI enforce one rule; two
    copies of a numeric bound drift. The message keeps the flag wording because
    tests and users both read it as a CLI diagnostic.
    """
    if (not 0 < min_chunk_s < max_chunk_s < MAX_CHUNK_LIMIT_S
            or max_chunk_s < 2 * min_chunk_s):
        raise ConfigError(
            "--max-chunk-s must be < 30 and > --min-chunk-s, and at least 2x "
            f"--min-chunk-s (got {min_chunk_s:g} and {max_chunk_s:g})")


def validate_target_s(target_s: float, min_chunk_s: float,
                      max_chunk_s: float) -> None:
    """Raise ConfigError unless the live target chunk length fits the bounds."""
    if not min_chunk_s <= target_s <= max_chunk_s:
        raise ConfigError(
            "--target-s must be between --min-chunk-s and --max-chunk-s "
            f"(got {target_s:g}, {min_chunk_s:g} and {max_chunk_s:g})")
```

- [ ] **Step 4: Delete the constant from `cli.py`**

Remove lines 62-63 of `omnilingual/cli.py`:

```python
# Sarvam's synchronous speech-to-text endpoint rejects audio of 30 seconds or more.
MAX_CHUNK_LIMIT_S = 30.0
```

Add `validate_chunk_bounds` and `validate_target_s` to the existing `from omnilingual.config import (...)` block in `cli.py`.

- [ ] **Step 5: Replace the transcribe check**

Replace:

```python
    if not 0 < min_chunk_s < max_chunk_s < MAX_CHUNK_LIMIT_S or max_chunk_s < 2 * min_chunk_s:
        _fail("--max-chunk-s must be < 30 and > --min-chunk-s, and at least 2x --min-chunk-s")
```

with:

```python
    try:
        validate_chunk_bounds(min_chunk_s, max_chunk_s)
    except ConfigError as exc:
        _fail(str(exc))
```

Keep the comment directly above it (`# Bad chunk bounds would only surface after normalizing...`).

- [ ] **Step 6: Replace the live checks**

Replace:

```python
    if not 0 < min_chunk_s < max_chunk_s < MAX_CHUNK_LIMIT_S or max_chunk_s < 2 * min_chunk_s:
        _fail("--max-chunk-s must be < 30 and > --min-chunk-s, and at least 2x --min-chunk-s")
    if not min_chunk_s <= target_s <= max_chunk_s:
        _fail("--target-s must be between --min-chunk-s and --max-chunk-s")
```

with:

```python
    try:
        validate_chunk_bounds(min_chunk_s, max_chunk_s)
        validate_target_s(target_s, min_chunk_s, max_chunk_s)
    except ConfigError as exc:
        _fail(str(exc))
```

Leave the following `if stt_workers < 1:` block untouched.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/test_cli.py tests/test_cli_live.py -q`
Expected: PASS. `tests/test_cli_live.py` line 78 asserts `"--max-chunk-s must be < 30" in res.output`; that test passing is the proof the extraction was behaviour-preserving.

- [ ] **Step 8: Commit**

```bash
git add omnilingual/config.py omnilingual/cli.py tests/test_config.py
git commit -m "refactor: extract shared chunk-bound validators into config"
```

---

### Task 2: Add the `ui` extra and the console script

Every later task needs the web dependencies importable, so add them now. This also creates the package directory the entry point targets (the entry point itself becomes valid in Task 9).

**Files:**
- Modify: `pyproject.toml`
- Create: `omnilingual/ui/__init__.py`
- Test: `tests/test_ui_packaging.py`

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `omnilingual.ui`; console script `omnilingual-ui` → `omnilingual.ui.__main__:main` (target file arrives in Task 9); extra `ui` installable via `uv sync --extra ui`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ui_packaging.py`:

```python
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def test_ui_extra_declares_the_three_web_dependencies():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "ui" in extras, "the ui extra must exist so the CLI gains no web dep"
    joined = " ".join(extras["ui"])
    assert "fastapi" in joined
    assert "uvicorn" in joined
    assert "pywebview" in joined


def test_console_script_points_at_the_ui_entry_point():
    scripts = _pyproject()["project"]["scripts"]
    assert scripts["omnilingual-ui"] == "omnilingual.ui.__main__:main"


def test_ui_package_is_not_imported_by_the_cli():
    # The CLI must never acquire a web dependency, so nothing outside ui/ may
    # reach omnilingual.ui on its import path.
    offenders = [p.name for p in (REPO / "omnilingual").rglob("*.py")
                 if p.parent.name != "ui"
                 and "omnilingual.ui" in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_ui_package_imports_without_web_dependencies():
    import omnilingual.ui  # noqa: F401
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_ui_packaging.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'omnilingual.ui'`.

- [ ] **Step 3: Add the extra and script to `pyproject.toml`**

Under `[project.optional-dependencies]`, alongside the existing `local-stt` / `diarize` / `local-mt` entries, add:

```toml
ui = ["fastapi>=0.115", "uvicorn[standard]>=0.32", "pywebview>=5.0"]
```

Under `[project.scripts]`, alongside the existing `omnilingual` entry, add:

```toml
omnilingual-ui = "omnilingual.ui.__main__:main"
```

`uvicorn[standard]` supplies the WebSocket implementation, so there is no separate `websockets` pin. Do not touch `[project] dependencies`.

- [ ] **Step 4: Create the package**

Create `omnilingual/ui/__init__.py`:

```python
"""Local desktop-style UI over the omnilingual pipeline.

Imported by nothing in the CLI: this package, and the FastAPI/pywebview
dependencies behind the `ui` extra, must never sit on the CLI's import path or
`omnilingual transcribe` would acquire a web dependency.
"""
```

- [ ] **Step 5: Install the extra**

Run: `uv sync --extra ui`
Expected: resolves and installs `fastapi`, `uvicorn`, `pywebview` and their transitive dependencies. `uv.lock` updates.

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/test_ui_packaging.py -q`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock omnilingual/ui/__init__.py tests/test_ui_packaging.py
git commit -m "build: add ui extra and omnilingual-ui console script"
```

---

### Task 3: Make `run_live` drivable from a worker thread and emit structured rows

This is the one change to existing code. Two gaps block the UI: `run_live` installs a `SIGINT` handler unconditionally, and `signal.signal()` raises `ValueError` off the main thread; and it offers no way to hand transcript text to a caller, only pre-rendered status strings, which deliberately contain no transcript text.

**Files:**
- Modify: `omnilingual/pipeline/live.py` (stdlib imports; signature at 147-150; line 210; lines 369-383; lines 426-434; line 450)
- Test: `tests/pipeline/test_run_live.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  ```python
  def run_live(opts: LiveOptions, settings, stt, translator, *,
               diarizer=None,
               status: Callable[[str], None] | None = None,
               capture_factory: Callable = LiveCapture,
               on_segment: Callable[[Segment, bool], None] | None = None,
               stop_event: threading.Event | None = None) -> int:
  ```
  - `on_segment(seg, keep)` — called once per emitted segment, **after** `keep` is computed and **before** the writers are fed. Fires for dropped silence segments too, with `keep=False`. Exceptions raised by the callback are logged and swallowed, never propagated.
  - `stop_event` — when supplied, becomes the run's own stop flag; setting it stops the run exactly as the first Ctrl+C press does.
  - Both default to `None`, so all 12 existing call sites in `cli.py` and both existing test modules are unaffected.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pipeline/test_run_live.py`:

```python
@respx.mock
def test_on_segment_fires_in_order_with_keep_flag(respx_mock, tmp_path):
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    seen = []
    code = run_live(_opts(tmp_path, stt_workers=1), settings, SarvamSTT(settings),
                    MayuraTranslator(settings), status=lambda m: None,
                    capture_factory=factory,
                    on_segment=lambda seg, keep: seen.append((seg.text, keep)))
    assert code == 0
    assert len(seen) >= 3
    # Order is the contract: rows must arrive in transcript order, not in
    # worker completion order.
    texts = [t for t, _ in seen]
    assert texts.index("Bravo") < texts.index("Hello") < texts.index("Vanakkam")
    # Every scripted chunk here carries speech, so all are kept.
    assert all(keep for _, keep in seen)


@respx.mock
def test_on_segment_reports_every_sealed_chunk(respx_mock, tmp_path):
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    seen = []
    run_live(_opts(tmp_path, stt_workers=1), settings, SarvamSTT(settings),
             MayuraTranslator(settings), status=lambda m: None,
             capture_factory=factory,
             on_segment=lambda seg, keep: seen.append((seg.status, keep)))
    # Whatever the ok/silence mix, the callback must be told about every sealed
    # chunk — including any silence the writer drops — or a live view freezes
    # during a quiet stretch and looks broken.
    assert seen, "on_segment must fire at least once"
    assert all(isinstance(keep, bool) for _, keep in seen)


@respx.mock
def test_on_segment_exception_does_not_end_the_run(respx_mock, tmp_path):
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    calls = []

    def boom(seg, keep):
        calls.append(1)
        raise RuntimeError("ui rendering bug")

    code = run_live(_opts(tmp_path, stt_workers=1), settings, SarvamSTT(settings),
                    MayuraTranslator(settings), status=lambda m: None,
                    capture_factory=factory, on_segment=boom)
    assert calls, "the raising callback must actually have been called"
    # A UI bug must not kill a meeting that is being transcribed.
    assert code == 0
    text = (tmp_path / "meeting.md").read_text(encoding="utf-8")
    assert "Bravo" in text and "Vanakkam" in text


@respx.mock
def test_stop_event_stops_the_run(respx_mock, tmp_path):
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    stop = threading.Event()
    seen = []

    def on_segment(seg, keep):
        seen.append(seg)
        stop.set()  # stop after the first row, like a user pressing Stop

    code = run_live(_opts(tmp_path, stt_workers=1), settings, SarvamSTT(settings),
                    MayuraTranslator(settings), status=lambda m: None,
                    capture_factory=factory,
                    on_segment=on_segment, stop_event=stop)
    assert stop.is_set()
    assert len(seen) < 3, "stopping after the first row must cut the run short"
    # Whatever was captured is still a valid, finalized file.
    assert (tmp_path / "meeting.md").exists()
    assert "· growing" not in (tmp_path / "meeting.md").read_text(encoding="utf-8")


@respx.mock
def test_run_live_works_off_the_main_thread(respx_mock, tmp_path):
    # The regression test for the SIGINT guard: signal.signal() raises
    # ValueError off the main thread, which is exactly how the UI calls it.
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    box = {}

    def worker():
        try:
            box["code"] = run_live(
                _opts(tmp_path, stt_workers=1), settings, SarvamSTT(settings),
                MayuraTranslator(settings), status=lambda m: None,
                capture_factory=factory,
                on_segment=lambda seg, keep: None)
        except BaseException as exc:  # noqa: BLE001 - surfaced in the assert
            box["error"] = exc

    th = threading.Thread(target=worker)
    th.start()
    th.join(timeout=120)
    assert not th.is_alive(), "run_live hung when driven from a worker thread"
    assert "error" not in box, f"run_live raised off the main thread: {box.get('error')!r}"
    assert box["code"] == 0
```

Add `import threading` to the stdlib import block at the top of `tests/pipeline/test_run_live.py` (it currently starts with `import re`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_run_live.py -q -k "on_segment or stop_event or main_thread"`
Expected: FAIL. `test_run_live_works_off_the_main_thread` fails with `ValueError: signal only works in main thread of the main interpreter`; the others fail on the unexpected keyword argument.

- [ ] **Step 3: Add the logger to `pipeline/live.py`**

`live.py` has no `logging` import today. Add `import logging` to its stdlib import block (after `import json`, before `import os`), and add directly after the import block:

```python
log = logging.getLogger(__name__)
```

- [ ] **Step 4: Extend the signature**

Replace:

```python
def run_live(opts: LiveOptions, settings, stt, translator, *,
             diarizer=None,
             status: Callable[[str], None] | None = None,
             capture_factory: Callable = LiveCapture) -> int:
    """Run a live session. Returns the process exit code (0/1/2)."""
    say = status or (lambda msg: None)
```

with:

```python
def run_live(opts: LiveOptions, settings, stt, translator, *,
             diarizer=None,
             status: Callable[[str], None] | None = None,
             capture_factory: Callable = LiveCapture,
             on_segment: Callable[[Segment, bool], None] | None = None,
             stop_event: threading.Event | None = None) -> int:
    """Run a live session. Returns the process exit code (0/1/2).

    on_segment(seg, keep) is the structured counterpart to the pre-rendered
    `status` strings: it carries the transcript text a GUI needs to render a
    row. It fires for every sealed chunk, including silence the writers drop, so
    a live view never appears frozen during a quiet stretch. A raising callback
    is logged and swallowed — UI code must not be able to end the session.
    stop_event lets a host that owns SIGINT (a GUI server) stop the run without
    sending a signal.
    """
    say = status or (lambda msg: None)
```

- [ ] **Step 5: Adopt the caller's stop flag**

At line 210, replace:

```python
    stop = threading.Event()
```

with:

```python
    stop = stop_event or threading.Event()
```

`capture_loop` already tests `stop.is_set()`, so it honours the flag unchanged and the graceful-shutdown path is untouched.

- [ ] **Step 6: Guard SIGINT registration**

Replace lines 369-383:

```python
    # Two-stage Ctrl+C: first stops gracefully, second reaps ffmpeg and exits.
    sigints = 0
    prev = signal.getsignal(signal.SIGINT)

    def on_sigint(signum, frame):
        nonlocal sigints
        sigints += 1
        if sigints == 1:
            say(f"[live {elapsed()}] stopping… (Ctrl+C again to quit now)")
            stop.set()
        else:
            capture.close()
            os._exit(2)

    signal.signal(signal.SIGINT, on_sigint)
```

with:

```python
    # Two-stage Ctrl+C: first stops gracefully, second reaps ffmpeg and exits.
    # Only a main-thread host can install a handler — signal.signal() raises
    # ValueError otherwise — and a GUI server owns SIGINT itself, stopping the
    # run through stop_event instead. getsignal() is safe from any thread, so
    # only the registration is conditional.
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

- [ ] **Step 7: Fire `on_segment` after `keep` is computed**

Replace lines 426-434:

```python
            seg, delta, _ = item
            say(status_line(state["next"] - 1, seg))
            keep = True
            if (seg.status == "no_speech"
                    and seg.chunk.end_s - state["last_kept_end"] <= 30.0):
                keep = False
            if keep:
                for w in writers:
                    w.append_segment(seg, delta)
```

with:

```python
            seg, delta, _ = item
            say(status_line(state["next"] - 1, seg))
            keep = True
            if (seg.status == "no_speech"
                    and seg.chunk.end_s - state["last_kept_end"] <= 30.0):
                keep = False
            if on_segment is not None:
                # After `keep` is known so the caller can mirror the file, and
                # before the writers so a slow consumer cannot delay capture.
                try:
                    on_segment(seg, keep)
                except Exception:  # noqa: BLE001 - a UI bug must not end the run
                    log.exception("on_segment callback failed")
            if keep:
                for w in writers:
                    w.append_segment(seg, delta)
```

- [ ] **Step 8: Guard the restore**

In the `finally` block at line 450, replace:

```python
    finally:
        signal.signal(signal.SIGINT, prev)
```

with:

```python
    finally:
        if owns_sigint:
            signal.signal(signal.SIGINT, prev)
```

Guarding the restore matters as much as the registration: an unguarded restore in a thread that never installed the handler would raise and mask the real exit path.

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest tests/pipeline/ -q`
Expected: PASS, including the 5 new tests and all pre-existing ones in `test_run_live.py` and `test_recovery.py`.

- [ ] **Step 10: Run the whole suite to confirm no CLI regression**

Run: `uv run pytest -q`
Expected: PASS. The 12 `cli.py` call sites are unaffected because both new params are keyword-only with `None` defaults, and the CLI's two-stage Ctrl+C still works because it runs on the main thread.

- [ ] **Step 11: Commit**

```bash
git add omnilingual/pipeline/live.py tests/pipeline/test_run_live.py
git commit -m "feat: add on_segment and stop_event to run_live, guard SIGINT off-thread"
```

---

### Task 4: Settings store and `.env` secret handling

Two leaf modules with no dependencies on the rest of the UI. `settings.py` is the run-preference store; `secrets.py` reports which API keys exist and writes them to the repo's gitignored `.env`, never revealing a value.

**Files:**
- Create: `omnilingual/ui/settings.py`
- Create: `omnilingual/ui/secrets.py`
- Test: `tests/ui/test_settings.py`, `tests/ui/test_secrets.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `settings.DEFAULTS: dict[str, object]` — the spec's §9 parameter table.
  - `settings.config_dir() -> Path`, `settings.settings_path() -> Path`
  - `settings.load() -> dict[str, object]` — defaults merged under whatever is stored; never raises on a missing or corrupt file; never creates the file.
  - `settings.save(values) -> dict[str, object]` — rejects unknown keys with `ValueError`; writes atomically at mode `0600` inside a `0700` directory.
  - `secrets.KEYS: tuple[str, ...]` — `("SARVAM_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY")`.
  - `secrets.env_path() -> Path` — the repo `.env`; overridable via `OMNILINGUAL_ENV_FILE`.
  - `secrets.present() -> dict[str, bool]`
  - `secrets.set_key(name: str, value: str) -> None` — inserts or replaces **in place**, preserving comments and unrelated keys.
  - `secrets.clear(name: str) -> None` — no-op when absent.

- [ ] **Step 1: Write the failing settings tests**

Create `tests/ui/test_settings.py`:

```python
import stat

import pytest

from omnilingual.ui import settings


def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    return settings


def test_defaults_mirror_the_cli_flag_surface(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    assert settings.load() == settings.DEFAULTS


def test_defaults_cover_every_panel_control():
    for key in ("mode", "source", "out", "stt", "stt_model", "mt", "mt_model",
                "diarize", "num_speakers", "langs", "english_only", "work_dir",
                "device", "mic_only", "target_s", "max_chunk_s", "min_chunk_s",
                "noise_db", "stt_workers", "max_cost"):
        assert key in settings.DEFAULTS, f"{key} missing from DEFAULTS"


def test_save_then_load_round_trips(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    settings.save({"out": "/tmp/notes.md", "stt": "groq", "num_speakers": 4})
    loaded = settings.load()
    assert loaded["out"] == "/tmp/notes.md"
    assert loaded["stt"] == "groq"
    assert loaded["num_speakers"] == 4
    # Untouched keys keep their defaults rather than becoming None.
    assert loaded["device"] == "Omnilingual"


def test_save_rejects_unknown_keys(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    with pytest.raises(ValueError) as exc:
        settings.save({"nope": 1})
    assert "nope" in str(exc.value)


def test_save_never_writes_an_unknown_key(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        settings.save({"out": "a.md", "nope": 1})
    assert "nope" not in settings.settings_path().read_text(encoding="utf-8")


def test_settings_file_is_private(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    settings.save({"out": "a.md"})
    path = settings.settings_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(settings.config_dir().stat().st_mode) == 0o700


def test_load_falls_back_to_defaults_on_corrupt_file(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    settings.save({"out": "a.md"})
    settings.settings_path().write_text("this is not = valid toml [[[",
                                        encoding="utf-8")
    assert settings.load() == settings.DEFAULTS


def test_load_does_not_create_a_file(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    settings.load()
    assert not settings.settings_path().exists()
```

- [ ] **Step 2: Write the failing secrets tests**

Create `tests/ui/test_secrets.py`:

```python
import stat

import pytest

from omnilingual.ui import secrets

FIXTURE = """# Omnilingual secrets — gitignored. Used via: uv run --env-file .env
# Groq: console.groq.com/keys
GROQ_API_KEY=groq-existing
# Gemini: aistudio.google.com/apikey
GEMINI_API_KEY=gemini-existing
"""


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNILINGUAL_ENV_FILE", str(tmp_path / ".env"))
    path = secrets.env_path()
    path.write_text(FIXTURE, encoding="utf-8")
    return path


def test_env_path_is_the_repo_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNILINGUAL_ENV_FILE", raising=False)
    path = secrets.env_path()
    assert path.name == ".env"
    assert path.parent.name == "omnilingual"


def test_present_reports_booleans_only(env_file):
    result = secrets.present()
    assert result == {"SARVAM_API_KEY": False, "GROQ_API_KEY": True,
                      "GEMINI_API_KEY": True}
    assert all(isinstance(v, bool) for v in result.values())


def test_set_key_appends_and_preserves_everything(env_file):
    secrets.set_key("SARVAM_API_KEY", "sarvam-new")
    text = env_file.read_text(encoding="utf-8")
    assert "sarvam-new" in text
    assert "GROQ_API_KEY=groq-existing" in text
    assert "GEMINI_API_KEY=gemini-existing" in text
    assert text.count("# Omnilingual secrets") == 1
    assert text.count("# Groq:") == 1


def test_set_key_replaces_in_place_without_duplicating(env_file):
    secrets.set_key("GROQ_API_KEY", "groq-replaced")
    text = env_file.read_text(encoding="utf-8")
    assert text.count("GROQ_API_KEY=") == 1
    assert "GROQ_API_KEY=groq-replaced" in text
    assert "groq-existing" not in text


def test_set_key_rejects_a_name_outside_the_known_set(env_file):
    with pytest.raises(ValueError):
        secrets.set_key("AWS_SECRET_ACCESS_KEY", "x")
    assert "AWS_SECRET" not in env_file.read_text(encoding="utf-8")


def test_set_key_keeps_mode_0600(env_file):
    env_file.chmod(0o600)
    secrets.set_key("SARVAM_API_KEY", "v")
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600


def test_clear_removes_the_line_and_keeps_others(env_file):
    secrets.clear("GROQ_API_KEY")
    text = env_file.read_text(encoding="utf-8")
    assert "GROQ_API_KEY" not in text
    assert "GEMINI_API_KEY=gemini-existing" in text
    assert text.count("# Gemini:") == 1


def test_clear_is_a_no_op_when_absent(env_file):
    before = env_file.read_text(encoding="utf-8")
    secrets.clear("SARVAM_API_KEY")
    assert env_file.read_text(encoding="utf-8") == before


def test_set_key_creates_the_file_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNILINGUAL_ENV_FILE", str(tmp_path / "fresh" / ".env"))
    secrets.set_key("GEMINI_API_KEY", "g")
    path = secrets.env_path()
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "GEMINI_API_KEY=g" in path.read_text(encoding="utf-8")
```

- [ ] **Step 3: Run both test files to verify they fail**

Run: `uv run pytest tests/ui/ -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'omnilingual.ui.settings'`.

- [ ] **Step 4: Write `omnilingual/ui/settings.py`**

```python
"""Run preferences for the UI, persisted outside the repo.

Deliberately separate from the repo's .env (see secrets.py) and from the CLI:
phase 1 does not make the CLI read this file. It lives beside setup-mac.conf
with the same permissions, so a preferences file is never group- or
world-readable.
"""

from __future__ import annotations

import os
import stat
import tempfile
import tomllib
from pathlib import Path

# One entry per control in the spec's parameter table. Defaults match the CLI's
# own defaults exactly, so the panel opens on the configuration the command line
# would have used.
DEFAULTS: dict[str, object] = {
    "mode": "live",
    "source": "",
    "out": "standup.md",
    "stt": "sarvam",
    "stt_model": "",
    "mt": "mayura",
    "mt_model": "",
    "diarize": False,
    "num_speakers": 3,
    "langs": [],
    "english_only": False,
    "work_dir": "",
    "device": "Omnilingual",
    "mic_only": False,
    "target_s": 8.0,
    "max_chunk_s": 28.0,
    "min_chunk_s": 5.0,
    "noise_db": -35.0,
    "stt_workers": 2,
    "max_cost": 50.0,
}


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "omnilingual"


def settings_path() -> Path:
    return config_dir() / "ui.toml"


def load() -> dict[str, object]:
    """Return the defaults merged under the stored file.

    Never raises: an unreadable or corrupt store must not stop the app from
    opening, so the user just gets defaults and can re-save over the damage.
    """
    values = dict(DEFAULTS)
    try:
        stored = tomllib.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return values
    for key, value in stored.items():
        if key in DEFAULTS:  # ignore junk keys rather than importing them
            values[key] = value
    return values


def _fmt(key: str, value: object) -> str:
    if isinstance(value, bool):
        return f"{key} = {'true' if value else 'false'}"
    if isinstance(value, (int, float)):
        return f"{key} = {value!r}"
    if isinstance(value, list):
        return f"{key} = {[str(v) for v in value]!r}"
    return f"{key} = {str(value)!r}"


def save(values) -> dict[str, object]:
    """Validate and persist atomically. Returns the stored state.

    Validation is by rejection, not coercion: an unknown key is almost always a
    renamed control, and silently dropping it would look like the UI forgot the
    setting.
    """
    unknown = sorted(set(values) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown setting(s): {', '.join(unknown)}")
    merged = load()
    merged.update(values)

    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, stat.S_IRWXU)

    body = "\n".join(_fmt(key, merged[key]) for key in DEFAULTS) + "\n"

    # Write a sibling, then rename: a crash mid-write must not leave a
    # truncated store that load() would silently discard.
    handle, tmp = tempfile.mkstemp(dir=str(directory), prefix="ui-", suffix=".toml")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(tmp, settings_path())
    except BaseException:
        os.unlink(tmp)
        raise
    return merged
```

- [ ] **Step 5: Write `omnilingual/ui/secrets.py`**

```python
"""Which API keys exist, and writing them to the repo's gitignored .env.

A value never leaves this module: present() answers with booleans, and nothing
here echoes, logs, or passes a value to a subprocess, so it cannot appear in
`ps`. The .env line semantics deliberately match scripts/setup-mac.sh — keep
comments and unrelated keys, replace a key in place rather than appending a
duplicate, mode 0600, write atomically. That shell implementation cannot be
imported, so this is a cross-language duplication of a small invariant set; if
one changes, change both.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
from pathlib import Path

KEYS: tuple[str, ...] = ("SARVAM_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY")

REPO_ROOT = Path(__file__).resolve().parents[2]


def env_path() -> Path:
    """The .env the CLI itself reads. Overridable so tests need no repo."""
    override = os.environ.get("OMNILINGUAL_ENV_FILE")
    if override:
        return Path(override)
    return REPO_ROOT / ".env"


def _assign(name: str) -> re.Pattern[str]:
    # Tolerate `export NAME=`, surrounding whitespace, and any quoting.
    return re.compile(r"^\s*(?:export\s+)?" + re.escape(name) + r"\s*=\s*(.*?)\s*$")


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _lines() -> list[str]:
    try:
        return env_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def present() -> dict[str, bool]:
    """A boolean per known key. Never the values themselves."""
    lines = _lines()
    found: dict[str, bool] = {}
    for name in KEYS:
        pattern = _assign(name)
        found[name] = any(
            match and _unquote(match.group(1))
            for line in lines if (match := pattern.match(line))
        )
    return found


def set_key(name: str, value: str) -> None:
    """Insert or replace one key, in place, leaving every other line alone."""
    if name not in KEYS:
        raise ValueError(f"unknown key: {name}")
    lines = _lines()
    pattern = _assign(name)
    for index, line in enumerate(lines):
        if pattern.match(line):
            lines[index] = f"{name}={value}"
            break
    else:
        lines.append(f"{name}={value}")
    _write(lines)


def clear(name: str) -> None:
    """Remove a key's line. A no-op when the key is not there."""
    if name not in KEYS:
        raise ValueError(f"unknown key: {name}")
    lines = _lines()
    pattern = _assign(name)
    kept = [line for line in lines if not pattern.match(line)]
    if len(kept) != len(lines):
        _write(kept)


def _write(lines: list[str]) -> None:
    path = env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".env-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/ui/ -q`
Expected: PASS (17 tests).

- [ ] **Step 7: Commit**

```bash
git add omnilingual/ui/settings.py omnilingual/ui/secrets.py tests/ui/
git commit -m "feat(ui): add settings store and .env secret handling"
```

---

### Task 5: Audio readiness probe and fix actions

Wraps the existing `scripts/audio-devices.swift` helper, which `setup-mac.sh` already compiles and drives, and adds the one thing the shell script cannot do from a GUI: an empirical microphone-permission probe.

**Files:**
- Create: `omnilingual/ui/audio.py`
- Test: `tests/ui/test_audio.py`

**Interfaces:**
- Consumes: `scripts/audio-devices.swift`; `shutil.which`; `subprocess`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class Readiness:
      device: bool
      blackhole: bool
      ffmpeg: bool
      ffprobe: bool
      mic_authorized: bool
      output: str | None
      detail: list[str]
      @property
      def ok(self) -> bool: ...
      def as_dict(self) -> dict: ...

  def helper_path() -> Path             # compiles audio-devices.swift on demand
  def device_list() -> str              # stdout of the helper's `list`
  def _mic_authorized(device: str) -> bool
  def probe(device: str = "Omnilingual", *, mic: bool = True) -> Readiness
  def setup(device: str = "Omnilingual") -> Iterator[str]   # yields progress lines
  def restart_daemon() -> None          # via the macOS authorisation prompt
  ```

- [ ] **Step 1: Write the failing tests**

Create `tests/ui/test_audio.py`:

> **Authoritative sample.** The block below is `tests/ui/test_audio.py` verbatim as
> shipped at `302de17` — 33 test functions, 34 cases (one is parametrised). It
> supersedes the 10-test list this step originally prescribed: those 10 are still
> present and still pass, but review found three defects they did not catch, and
> 23 further tests now guard them. If this block and the shipped file ever
> disagree, the shipped file wins — do not copy a subset of it.
>

```python
import subprocess

import pytest

from omnilingual.ui import audio


@pytest.fixture
def no_sudo(monkeypatch):
    """Fail loudly if any test ever reaches a real privileged command."""
    def boom(*a, **k):
        raise AssertionError("test attempted a privileged/real command")
    monkeypatch.setattr(subprocess, "run", boom)


def _ready(**over):
    fields = dict(device=True, blackhole=True, ffmpeg=True, ffprobe=True,
                  mic_authorized=True, output="Speakers", detail=[])
    fields.update(over)
    return audio.Readiness(**fields)


# What the helper's `list` prints once setup() has run, in its real "uid | name"
# shape rather than the bare names the brief's own tests use.
_FULL_LISTING = ("BlackHole_2ch | BlackHole 2ch\n"
                 "omnilingual.aggregate | Omnilingual\n"
                 "BuiltInMicDevice | MacBook Pro Microphone\n")


def test_readiness_ok_requires_all_five_signals():
    assert _ready().ok is True
    assert _ready().as_dict()["ok"] is True


def test_readiness_not_ok_when_the_device_is_missing():
    assert _ready(device=False).ok is False


def test_readiness_not_ok_when_mic_permission_is_denied():
    assert _ready(mic_authorized=False).ok is False


def test_as_dict_shape_matches_the_spec():
    assert _ready(output="Multi-Output Device").as_dict() == {
        "ok": True, "device": "Omnilingual", "blackhole": True, "ffmpeg": True,
        "ffprobe": True, "mic_authorized": True,
        "output": "Multi-Output Device", "detail": [],
    }


def test_probe_reports_missing_binaries_without_running_anything(monkeypatch, no_sudo):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    result = audio.probe(mic=False)
    assert result.ffmpeg is False and result.ffprobe is False
    assert any("ffmpeg" in line for line in result.detail)


def test_probe_never_rewrites_the_system_output(monkeypatch):
    """Switching output silently breaks the volume keys, so probe is read-only."""
    calls = []

    def fake_run(cmd, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Omnilingual\nBlackHole 2ch\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "device_list", lambda: "Omnilingual\nBlackHole 2ch\n")
    monkeypatch.setattr(audio, "_current_output", lambda: "Speakers")
    audio.probe(device="Omnilingual", mic=False)
    joined = [" ".join(cmd) for cmd in calls]
    assert not any("SwitchAudioSource" in cmd and "-s" in cmd for cmd in joined)


def test_mic_probe_treats_permission_denied_as_unauthorized(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="AVFoundation: Permission denied"))
    assert audio._mic_authorized("Omnilingual") is False


def test_mic_probe_rejects_permission_markers_even_on_exit_zero(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(
            cmd, 0, stdout="", stderr="Device not permitted"))
    assert audio._mic_authorized("Omnilingual") is False


def test_mic_probe_accepts_a_clean_capture(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    assert audio._mic_authorized("Omnilingual") is True


def test_restart_daemon_uses_the_mac_authorisation_prompt(monkeypatch):
    seen = []

    def fake_run(cmd, **k):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    audio.restart_daemon()
    assert len(seen) == 1
    joined = " ".join(seen[0])
    assert "killall coreaudiod" in joined
    assert "administrator privileges" in joined


# --- below: coverage the brief's list leaves open.  Same rules, narrower claims.


def test_probe_consults_switchaudiosource_read_only(monkeypatch):
    """Stronger form of the read-only claim: probe really does run a command.

    The brief's version stubs `_current_output` away, so it cannot fail even if
    probe started setting the output.  Here only `_current_output`'s own internals
    are left intact, so the assertion is on a command that was actually issued.
    """
    calls = []

    def fake_run(cmd, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Speakers", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "device_list", lambda: "omnilingual.aggregate | Omnilingual\n")
    result = audio.probe(device="Omnilingual", mic=False)
    assert result.output == "Speakers"
    switches = [cmd for cmd in calls if "SwitchAudioSource" in cmd]
    assert switches, "probe should report the current output device"
    assert all("-c" in cmd and "-s" not in cmd for cmd in switches)


def test_current_output_is_none_when_switchaudiosource_is_absent(monkeypatch, no_sudo):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    assert audio._current_output() is None


def test_probe_does_not_claim_a_denial_it_never_observed(monkeypatch, no_sudo):
    """No device means no capture was attempted, so no denial may be reported.

    `no_sudo` turns any stray subprocess call into a failure, which is what makes
    "not probed" observable rather than asserted in prose.
    """
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "device_list", lambda: "BlackHole_2ch | BlackHole 2ch\n")
    monkeypatch.setattr(audio, "_current_output", lambda: "Speakers")
    result = audio.probe(device="Omnilingual", mic=True)
    assert result.device is False and result.blackhole is True
    assert result.mic_authorized is False
    assert not any("microphone" in line.lower() for line in result.detail)


def test_probe_reports_the_missing_device_and_blackhole(monkeypatch, no_sudo):
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "device_list", lambda: "BuiltInMicDevice | MacBook Pro Microphone\n")
    monkeypatch.setattr(audio, "_current_output", lambda: "Speakers")
    result = audio.probe(device="Omnilingual", mic=False)
    joined = " ".join(result.detail)
    assert "Omnilingual" in joined
    assert "brew install blackhole-2ch" in joined
    assert result.mic_authorized is True and not any(
        "microphone" in line.lower() for line in result.detail)


def test_probe_reports_the_mic_as_authorized_when_the_probe_is_skipped(monkeypatch):
    """mic=False means "not asked", so the answer must not be a false alarm.

    Task 6 returns probe(mic=False) from both fix endpoints, so flipping the
    unprobed branch to False would put "microphone access denied" in the banner
    of every *successful* setup while this suite stayed green.
    """
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "device_list", lambda: _FULL_LISTING)
    monkeypatch.setattr(audio, "_current_output", lambda: "Speakers")
    result = audio.probe(device="Omnilingual", mic=False)
    assert result.mic_authorized is True
    assert not any("microphone" in line.lower() for line in result.detail)


def test_probe_keeps_the_skipped_mic_authorized_even_with_no_device(monkeypatch, no_sudo):
    """Unprobed outranks "device missing": the branch order must not hide it."""
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "device_list", lambda: "")
    monkeypatch.setattr(audio, "_current_output", lambda: "Speakers")
    result = audio.probe(device="Omnilingual", mic=False)
    assert result.device is False
    assert result.mic_authorized is True


def test_probe_survives_a_helper_that_cannot_be_compiled(monkeypatch, no_sudo):
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "_current_output", lambda: None)

    def no_such_helper():
        raise FileNotFoundError("no swiftc")

    monkeypatch.setattr(audio, "device_list", no_such_helper)
    result = audio.probe(device="Omnilingual", mic=False)
    assert result.device is False and result.blackhole is False
    assert any("audio-device helper" in line for line in result.detail)


def test_mic_probe_captures_exactly_one_second_through_ffmpeg(monkeypatch):
    seen = {}

    def fake_run(cmd, **k):
        seen["cmd"] = cmd
        seen["kwargs"] = k
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert audio._mic_authorized("Omnilingual") is True
    assert seen["cmd"][0] == "ffmpeg"
    assert seen["cmd"][seen["cmd"].index("-f") + 1] == "avfoundation"
    assert seen["cmd"][seen["cmd"].index("-t") + 1] == "1"
    # The colon is load-bearing: a bare name asks avfoundation for a *video*
    # device and fails "Video device not found" on a machine whose mic is fine.
    assert seen["cmd"][seen["cmd"].index("-i") + 1] == ":Omnilingual"
    assert seen["kwargs"].get("shell") in (None, False)


def test_capture_verdict_points_at_privacy_settings_on_a_real_denial(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(
            cmd, 1, stdout="",
            stderr="[AVFoundation indev] Error opening input: Permission denied"))
    ok, reason = audio._capture_verdict("Omnilingual")
    assert ok is False
    assert "Privacy & Security" in reason


def test_capture_verdict_does_not_blame_permission_for_another_failure(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(
            cmd, 251, stdout="",
            stderr="[AVFoundation indev] Video device not found\n"
                   "Error opening input file :Omnilingual."))
    ok, reason = audio._capture_verdict("Omnilingual")
    assert ok is False
    assert "Privacy & Security" not in reason
    assert "Error opening input file" in reason


def test_probe_does_not_blame_permission_for_an_unrelated_capture_failure(monkeypatch):
    calls = []

    def fake_run(cmd, **k):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(
            cmd, 251, stdout="",
            stderr="[AVFoundation indev] Video device not found\n"
                   "Error opening input file :Omnilingual.")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "device_list", lambda: "omnilingual.aggregate | Omnilingual\n")
    monkeypatch.setattr(audio, "_current_output", lambda: "Speakers")
    result = audio.probe(device="Omnilingual", mic=True)
    assert result.mic_authorized is False
    joined = " ".join(result.detail)
    assert "Error opening input file" in joined
    assert "Privacy & Security" not in joined
    assert any(cmd[0] == "ffmpeg" for cmd in calls), "the capture must be attempted"


def test_probe_reports_authorized_when_the_capture_succeeds(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "device_list", lambda: _FULL_LISTING)
    monkeypatch.setattr(audio, "_current_output", lambda: "Speakers")
    result = audio.probe(device="Omnilingual", mic=True)
    assert result.mic_authorized is True
    assert result.detail == []


def test_permission_markers_are_only_strings_ffmpeg_actually_emits():
    """No speculative markers: each one here was seen in a real ffmpeg refusal.

    A marker that never appears can only ever misfire, and markers are tested
    before the exit code — so an invented one converts clean captures into
    "macOS denied microphone access". ffmpeg writes "[AVFoundation indev @ 0x…]",
    never "avfoundation:", so that third string was removed in the fix round.
    """
    assert audio._PERMISSION_MARKERS == ("permission denied", "not permitted")


@pytest.mark.parametrize("exc", [
    OSError("ffmpeg is not on PATH"),
    subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30),
])
def test_capture_verdict_explains_a_probe_it_never_ran(monkeypatch, exc):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(subprocess, "run", boom)
    ok, reason = audio._capture_verdict("Omnilingual")
    assert ok is False
    assert reason, "an unrun probe must still explain itself"
    assert "Privacy & Security" not in reason
    assert "permission" not in reason.lower()


def test_probe_survives_a_capture_that_cannot_be_run(monkeypatch):
    """A 500 on /api/audio is the opaque failure this module exists to prevent."""
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30)

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "device_list", lambda: _FULL_LISTING)
    monkeypatch.setattr(audio, "_current_output", lambda: "Speakers")
    result = audio.probe(device="Omnilingual", mic=True)
    assert result.mic_authorized is False
    joined = " ".join(result.detail)
    assert joined, "the failure must reach the page"
    assert "Privacy & Security" not in joined


def test_restart_daemon_interpolates_nothing_into_the_shell_script(monkeypatch):
    """A value placed inside the `do shell script` string would be executed as shell."""
    seen = []

    def fake_run(cmd, **k):
        seen.append((cmd, k))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    audio.restart_daemon()
    cmd, kwargs = seen[0]
    assert cmd[0] == "osascript" and cmd[1] == "-e"
    assert cmd[2] == 'do shell script "killall coreaudiod" with administrator privileges'
    assert len(cmd) == 3
    assert kwargs.get("shell") in (None, False)
    # A cancelled auth dialog exits nonzero; without check=True that would read
    # as success and the UI would claim the daemon restarted.
    assert kwargs.get("check") is True


def test_helper_path_compiles_the_swift_helper_once(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(audio, "HELPER_SRC", tmp_path / "audio-devices.swift")
    monkeypatch.setattr(audio, "_helper", None)
    monkeypatch.setattr(subprocess, "run", fake_run)
    src = tmp_path / "audio-devices.swift"
    src.write_text("// stub\n", encoding="utf-8")
    first = audio.helper_path()
    second = audio.helper_path()
    assert first == second, "the compiled helper must be cached per process"
    assert first.parent.is_dir() and first.name == "audio-devices"
    assert len(calls) == 1
    assert calls[0][0] == "swiftc"
    assert str(src) in calls[0]


def test_helper_path_refuses_to_build_a_helper_that_is_not_there(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "HELPER_SRC", tmp_path / "absent.swift")
    monkeypatch.setattr(audio, "_helper", None)

    def boom(*a, **k):
        raise AssertionError("must not compile a missing helper")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(FileNotFoundError):
        audio.helper_path()


def test_device_list_returns_the_helper_stdout(monkeypatch, tmp_path):
    binary = tmp_path / "audio-devices"
    monkeypatch.setattr(audio, "helper_path", lambda: binary)
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(
            cmd, 0, stdout="omnilingual.aggregate | Omnilingual\n", stderr=""))
    assert audio.device_list() == "omnilingual.aggregate | Omnilingual\n"


def test_setup_creates_the_aggregate_and_reports_ready(monkeypatch, tmp_path):
    binary = tmp_path / "audio-devices"
    calls = []

    def fake_run(cmd, **k):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(audio, "helper_path", lambda: binary)
    monkeypatch.setattr(audio, "device_list", lambda: "BlackHole_2ch | BlackHole 2ch\n")
    monkeypatch.setattr(audio.time, "sleep", lambda _s: None)
    monkeypatch.setattr(subprocess, "run", fake_run)
    lines = list(audio.setup())
    assert lines[0] == "building the audio-device helper"
    assert lines[-1] == "Aggregate Device 'Omnilingual' is ready"
    subcommands = [cmd[1] for cmd in calls if cmd[0] == str(binary)]
    assert subcommands == ["ensure", "check"], "ensure then verify, nothing else"
    assert not any(cmd[0] == "SwitchAudioSource" for cmd in calls), (
        "setup must never route system audio anywhere")


def test_setup_reports_a_helper_that_refuses_to_create_the_device(monkeypatch, tmp_path):
    binary = tmp_path / "audio-devices"
    monkeypatch.setattr(audio, "helper_path", lambda: binary)
    monkeypatch.setattr(audio, "device_list", lambda: "BlackHole_2ch | BlackHole 2ch\n")
    monkeypatch.setattr(audio.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(
            cmd, 1 if cmd[1:2] == ["ensure"] else 0, stdout="", stderr="no parts"))
    with pytest.raises(RuntimeError) as ei:
        list(audio.setup())
    assert "no parts" in str(ei.value)


def test_setup_gives_up_when_blackhole_never_appears(monkeypatch, tmp_path):
    slept = []
    monkeypatch.setattr(audio, "helper_path", lambda: tmp_path / "audio-devices")
    monkeypatch.setattr(audio, "device_list", lambda: "BuiltInMicDevice | Microphone\n")
    monkeypatch.setattr(audio.time, "sleep", slept.append)
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    with pytest.raises(RuntimeError) as ei:
        list(audio.setup())
    assert "brew install blackhole-2ch" in str(ei.value)
    assert len(slept) == 30, "bounded retry, not an unbounded hang"


def test_setup_refuses_a_device_the_helper_cannot_create(monkeypatch, tmp_path):
    """The helper hardcodes the name, so any other device would be a false success.

    audio-devices.swift creates and checks only "Omnilingual", so setup("Foo")
    would create Omnilingual, verify Omnilingual, and then report Foo as ready.
    """
    monkeypatch.setattr(audio, "helper_path", lambda: tmp_path / "audio-devices")

    def boom(*a, **k):
        raise AssertionError("must refuse before touching the helper")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(ValueError) as ei:
        list(audio.setup("Foo"))
    assert "Omnilingual" in str(ei.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/ui/test_audio.py -q`
Expected: FAIL — `ImportError: cannot import name 'audio' from 'omnilingual.ui'` (the module does not exist yet).

- [ ] **Step 3: Write `omnilingual/ui/audio.py`**

> **Authoritative sample, corrected — read this before lifting code from here.** The
> block below is `omnilingual/ui/audio.py` verbatim as shipped at `302de17` and
> unchanged since; this document round changed only this sample. An earlier revision
> was patched on the microphone-probe axis alone, which made the block *partly*
> correct and therefore more misleading than the originally stale version: a reader
> who trusted the corrected probe also inherited three defects review had already
> closed in the shipped module. All three are corrected here:
>
> - `setup()` refuses a device the Swift helper cannot create.
>   `scripts/audio-devices.swift:172` hardcodes `name: "Omnilingual"` and its
>   `check` (`:206`) tests only `"Omnilingual"` / `"Multi-Output Device"`, so
>   `setup("Foo")` created Omnilingual, verified Omnilingual, and then reported
>   `Aggregate Device 'Foo' is ready` — a false success from the one function whose
>   whole job is to tell the truth.
> - `setup()` also lost a redundant second `swiftc` compile: `helper_path()` had
>   already compiled the helper, and the block used to compile it again.
> - `_current_output()` wraps its `subprocess.run`, like its two sibling call sites.
>   A hung `SwitchAudioSource` would otherwise turn every `/api/audio` poll into a
>   500.
> - `device_list()`'s docstring states the helper's real `"<uid> | <name>"` output
>   instead of claiming bare names.
>
> The microphone-probe corrections from the previous round — the leading-colon
> `-i ":{device}"`, the two-marker `_PERMISSION_MARKERS`, and the `_capture_verdict`
> reason split — are unchanged and verified on real hardware. If this block and the
> shipped module ever disagree, the shipped module wins.
>

```python
"""Is audio capture ready, and the two ways to fix it when it is not.

Wraps scripts/audio-devices.swift — the same helper setup-mac.sh compiles and
drives — rather than reimplementing Aggregate Device creation. Everything here
is read-only except setup() and restart_daemon(), which the UI reaches only from
an explicit button. In particular nothing here ever switches the system output
device: the Multi-Output Device route silently breaks the volume keys, which is
why the current output is reported for diagnosis and never changed.

Stdlib plus omnilingual's own ffmpeg gate, deliberately: this package must stay
importable without the `ui` extra, because the CLI has no web dependency and
tests/test_ui_packaging.py fails any module here that reaches for one.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from omnilingual.audio.normalize import FfmpegMissingError, ensure_ffmpeg

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_SRC = REPO_ROOT / "scripts" / "audio-devices.swift"
DEVICE_NAME = "Omnilingual"
_TIMEOUT = 30

# How AVFoundation words a refusal, both seen in real ffmpeg output. Markers are
# tested before the exit code, so a string ffmpeg never emits can only ever turn
# a working capture into a denial: it spells "[AVFoundation indev @ 0x…]", never
# "avfoundation:", and that third guess was dropped rather than left to misfire.
_PERMISSION_MARKERS = ("permission denied", "not permitted")

_helper: Path | None = None


@dataclass(frozen=True)
class Readiness:
    device: bool
    blackhole: bool
    ffmpeg: bool
    ffprobe: bool
    mic_authorized: bool
    output: str | None
    detail: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all((self.device, self.blackhole, self.ffmpeg, self.ffprobe,
                    self.mic_authorized))

    def as_dict(self) -> dict:
        # "device" is the name here, while Readiness.device says whether it
        # exists: the page shows one and colours the other, and renaming either
        # would break the shape /api/audio already documents.
        return {
            "ok": self.ok,
            "device": DEVICE_NAME,
            "blackhole": self.blackhole,
            "ffmpeg": self.ffmpeg,
            "ffprobe": self.ffprobe,
            "mic_authorized": self.mic_authorized,
            "output": self.output,
            "detail": list(self.detail),
        }


def helper_path() -> Path:
    """Compile audio-devices.swift into a private temp binary, once per process.

    The temp directory is deliberately left behind: the binary is the process's
    working copy of the helper for as long as the UI runs.
    """
    global _helper
    if _helper is not None:
        return _helper
    if not HELPER_SRC.is_file():
        raise FileNotFoundError(f"missing {HELPER_SRC}")
    out = Path(tempfile.mkdtemp(prefix="omni-audio-")) / "audio-devices"
    subprocess.run(["swiftc", "-o", str(out), str(HELPER_SRC)], check=True,
                   capture_output=True, timeout=_TIMEOUT)
    _helper = out
    return out


def device_list() -> str:
    """Stdout of the helper's `list` subcommand.

    One `"<uid> | <name>"` line per audio device, so callers match a device by
    substring against the whole listing rather than parsing a column.
    """
    result = subprocess.run([str(helper_path()), "list"], capture_output=True,
                            text=True, timeout=_TIMEOUT)
    return result.stdout or ""


def _current_output() -> str | None:
    """The device the system plays through right now — reported, never changed.

    Best-effort by design: this is a diagnostic for "why is meeting audio not
    reaching BlackHole", so an absent or stuck SwitchAudioSource must not fail
    the readiness verdict the page is waiting on.
    """
    if shutil.which("SwitchAudioSource") is None:
        return None
    try:
        result = subprocess.run(["SwitchAudioSource", "-c"], capture_output=True,
                                text=True, timeout=_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout or "").strip() or None


def _short_error(text: str) -> str:
    """The last thing ffmpeg said, trimmed to something a page can show."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:200] if lines else "ffmpeg failed without a message"


def _capture_verdict(device: str) -> tuple[bool, str]:
    """Attempt one second of capture and report what actually happened.

    macOS exposes no supported way to read Microphone TCC status, so this is the
    only way to learn it. It works because capture runs through ffmpeg in this
    process rather than the browser's getUserMedia: there is no browser prompt,
    but macOS still refuses at the AVFoundation layer.

    The leading colon in the device string selects the audio section, which is
    the form live_capture.py opens capture with: a bare `-i Omnilingual` asks
    avfoundation for a *video* device of that name and fails "Video device not
    found", which says nothing about the microphone.

    Returns (authorized, reason). `reason` explains a failure and is empty when
    authorized, so the caller can name a permission denial only when one was
    actually observed. A probe that could not be run is also a failure with a
    reason, never a denial: the only subprocess call here that touches hardware,
    and the one most able to hang.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostdin", "-f", "avfoundation",
             "-i", f":{device}", "-t", "1", "-f", "null", "-"],
            capture_output=True, text=True, timeout=_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, _short_error(f"could not run the one-second capture: {exc}")
    stderr = result.stderr or ""
    if any(marker in stderr.lower() for marker in _PERMISSION_MARKERS):
        return False, ("macOS denied microphone access — approve this app in "
                       "System Settings → Privacy & Security → Microphone")
    if result.returncode != 0:
        # Not a denial: a busy device or a broken aggregate fails the same way.
        return False, _short_error(stderr + (result.stdout or ""))
    return True, ""


def _mic_authorized(device: str) -> bool:
    """Empirically test whether this process may record.

    Real audio I/O for one second, which is why probe() takes mic=False: a page
    poll that cannot ask must not make noise.
    """
    return _capture_verdict(device)[0]


def probe(device: str = DEVICE_NAME, *, mic: bool = True) -> Readiness:
    """Report capture readiness. Read-only, so safe to call at any time."""
    detail: list[str] = []
    ffmpeg = shutil.which("ffmpeg") is not None
    ffprobe = shutil.which("ffprobe") is not None
    if not (ffmpeg and ffprobe):
        # ensure_ffmpeg owns the wording of this failure for the CLI, so borrow
        # its message instead of keeping a second one to drift. probe reports
        # where the CLI raises, which is why it cannot simply call the gate.
        try:
            ensure_ffmpeg()
        except FfmpegMissingError as exc:
            detail.append(str(exc))

    listing = ""
    if ffmpeg and ffprobe:
        try:
            listing = device_list()
        except (OSError, subprocess.SubprocessError) as exc:
            detail.append(f"could not run the audio-device helper: {exc}")

    device_ok = device in listing
    blackhole = "BlackHole" in listing
    if listing and not device_ok:
        detail.append(f"the Aggregate Device '{device}' does not exist yet")
    if listing and not blackhole:
        detail.append("BlackHole 2ch is not installed; "
                      "run: brew install blackhole-2ch")

    mic_ok = False
    if not mic:
        mic_ok = True  # not probed, so do not report a false alarm
    elif ffmpeg and device_ok:
        mic_ok, reason = _capture_verdict(device)
        if not mic_ok:
            detail.append(f"microphone capture failed: {reason}")

    return Readiness(device=device_ok, blackhole=blackhole, ffmpeg=ffmpeg,
                     ffprobe=ffprobe, mic_authorized=mic_ok,
                     output=_current_output(), detail=detail)


def setup(device: str = DEVICE_NAME) -> Iterator[str]:
    """Create the Aggregate Device, yielding progress lines. Idempotent."""
    # audio-devices.swift hardcodes the name in both `ensure` and `check`, so any
    # other name would be created as Omnilingual, verified as Omnilingual, and
    # then reported as ready under the caller's name — a false success from the
    # one function whose whole job is to tell the truth.
    if device != DEVICE_NAME:
        raise ValueError(f"the helper only creates {DEVICE_NAME!r}")
    yield "building the audio-device helper"
    binary = helper_path()

    for attempt in range(30):
        try:
            if "BlackHole" in device_list():
                break
        except (OSError, subprocess.SubprocessError):
            pass
        if attempt == 0:
            yield ("waiting for BlackHole to appear; a fresh install may need "
                   "the audio daemon restarted")
        time.sleep(2)
    else:
        raise RuntimeError(
            "BlackHole never appeared. Install it with: "
            "brew install blackhole-2ch, then use 'Restart audio daemon' if "
            "this is a fresh Mac.")

    yield "creating the Aggregate Device"
    result = subprocess.run([str(binary), "ensure"], capture_output=True,
                            text=True, timeout=_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(f"device creation failed: {result.stderr.strip()}")
    subprocess.run([str(binary), "check"], capture_output=True, text=True,
                   timeout=_TIMEOUT, check=True)
    yield f"Aggregate Device '{device}' is ready"


def restart_daemon() -> None:
    """Restart coreaudiod through the standard macOS authorisation dialog.

    The UI has no TTY, so setup-mac.sh takes its no-sudo branch and skips this; a
    freshly installed BlackHole does not appear until coreaudiod restarts.

    The script string is a constant: anything interpolated between those quotes
    would be executed as shell by root. Nothing from a request reaches it.
    """
    subprocess.run(
        ["osascript", "-e",
         'do shell script "killall coreaudiod" with administrator privileges'],
        capture_output=True, text=True, timeout=120, check=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/ui/test_audio.py -q`
Expected: PASS (34 tests).

- [ ] **Step 5: Verify the real probe on this machine**

Run: `uv run python -c "import json; from omnilingual.ui import audio; print(json.dumps(audio.probe().as_dict(), indent=2))"`
Expected: a JSON object with all eight keys. Whether `ok` is true depends on whether this machine has the Aggregate Device configured; the point of the check is that the call returns a verdict instead of raising.

- [ ] **Step 6: Commit**

```bash
git add omnilingual/ui/audio.py tests/ui/test_audio.py
git commit -m "feat(ui): add audio readiness probe and setup actions"
```

---

### Task 6: `SessionRunner` — the only module that knows how a run executes

Owns a run's worker thread and turns pipeline callbacks into event dicts. It accepts an optional event loop so it can be unit-tested with no async machinery, and it knows nothing about HTTP.

**Files:**
- Create: `omnilingual/ui/session.py`
- Test: `tests/ui/test_session.py`

**Interfaces:**
- Consumes: `run_live`, `LiveOptions` (`omnilingual.pipeline.live`); `run`, `run_from_chunks`, `work_dir_for` (`omnilingual.pipeline`); `JsonCache` (`omnilingual.cache`); `render`, `render_english_only` (`omnilingual.render.markdown`); `Settings` (`omnilingual.config`).
- Produces:
  ```python
  class SessionRunner:
      def __init__(self, *, run_id: str, queue,
                   loop: asyncio.AbstractEventLoop | None = None) -> None: ...
      @staticmethod
      def segment_message(seq: int, seg, cost_inr: float, kept: bool) -> dict: ...
      def start_live(self, *, settings, stt, translator, diarizer,
                     opts: LiveOptions, capture_factory=LiveCapture) -> None: ...
      def start_recording(self, *, settings, stt, translator, diarizer,
                          source: Path, work_root: Path, out: Path,
                          english_only: bool = False) -> None: ...
      def recover(self, *, session_dir: Path, settings, stt, translator,
                  diarizer, out: Path, english_only: bool = False) -> None: ...
      def stop(self) -> None: ...
      def join(self, timeout: float | None = None) -> None: ...
      @property
      def status(self) -> str: ...          # idle|running|stopping|done|failed
      @property
      def session_dir(self) -> Path | None: ...
      @property
      def out_path(self) -> Path | None: ...
      @property
      def recoverable(self) -> bool: ...
      def snapshot(self) -> dict: ...       # the `state` message payload
  ```
  Events use the spec's §7.1 shapes: `{"type": "state"|"segment"|"status"|"log"|"end", ...}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/ui/test_session.py`:

```python
from pathlib import Path

import pytest
import respx

from omnilingual.config import load_settings
from omnilingual.models import Chunk, Segment
from omnilingual.pipeline.live import LiveOptions
from omnilingual.ui.session import SessionRunner

from tests.conftest import make_wav
from tests.pipeline.test_run_live import (  # reuse the proven seams
    FakeCapture,
    _blocks,
    _gaps,
    _pcm_3x20,
    _route,
)


class Collector:
    """An asyncio.Queue stand-in so the runner needs no event loop in tests."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def put_nowait(self, message: dict) -> None:
        self.items.append(message)

    def of(self, kind: str) -> list[dict]:
        return [m for m in self.items if m["type"] == kind]


def _providers():
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator
    config = load_settings(api_key="k")
    return config, SarvamSTT(config), MayuraTranslator(config)


def _live(tmp_path, capture_factory, **kw):
    queue = Collector()
    runner = SessionRunner(run_id="r1", queue=queue)
    opts = LiveOptions(out=tmp_path / "meeting.md", stt_workers=1, **kw)
    return runner, queue, opts, capture_factory


@respx.mock
def test_live_run_emits_state_segment_status_and_end(respx_mock, tmp_path):
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    config, stt, mt = _providers()
    runner, queue, opts, factory = _live(tmp_path, factory)

    runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                      opts=opts, capture_factory=factory)
    runner.join(timeout=120)

    assert runner.status == "done"
    kinds = [m["type"] for m in queue.items]
    assert kinds[0] == "state"
    assert "segment" in kinds
    assert "status" in kinds, "status lines must still reach the page"
    assert kinds[-1] == "end"

    seqs = [m["seq"] for m in queue.of("segment")]
    assert seqs == sorted(seqs), "segments must stream in transcript order"
    for message in queue.of("segment"):
        assert set(message) >= {"seq", "idx", "start_s", "end_s", "lang", "prob",
                                "text", "english", "status", "speaker", "kept",
                                "cost_inr"}
        assert isinstance(message["kept"], bool)

    end = queue.of("end")[0]
    assert end["exit_code"] == 0
    assert end["out"] == str(tmp_path / "meeting.md")
    assert end["recoverable"] is True
    assert Path(end["session_dir"]).is_dir()


@respx.mock
def test_live_run_reports_a_session_dir_for_recovery(respx_mock, tmp_path):
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    config, stt, mt = _providers()
    runner, queue, opts, factory = _live(tmp_path, factory)

    runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                      opts=opts, capture_factory=factory)
    runner.join(timeout=120)

    assert runner.session_dir is not None
    assert (runner.session_dir / "session.json").is_file()
    assert runner.recoverable is True


@respx.mock
def test_stop_leaves_a_finalized_file(respx_mock, tmp_path):
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    config, stt, mt = _providers()
    runner, queue, opts, factory = _live(tmp_path, factory)

    runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                      opts=opts, capture_factory=factory)
    runner.stop()
    runner.join(timeout=120)

    assert runner.status in {"stopping", "done"}
    assert (tmp_path / "meeting.md").exists()
    assert "· growing" not in (tmp_path / "meeting.md").read_text(encoding="utf-8")


@respx.mock
def test_recording_run_writes_a_rendered_transcript(respx_mock, tmp_path):
    _route(respx_mock)
    recording = tmp_path / "meeting.wav"
    make_wav(recording, [(0.0, 3.0)])

    config, stt, mt = _providers()
    queue = Collector()
    runner = SessionRunner(run_id="r2", queue=queue)
    out = tmp_path / "ui.md"

    runner.start_recording(settings=config, stt=stt, translator=mt, diarizer=None,
                           source=recording, work_root=tmp_path / "work", out=out)
    runner.join(timeout=300)

    assert runner.status == "done"
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "Transcript" in text
    assert queue.of("segment"), "batch progress must stream as segment messages"
    assert all(m["kept"] is True for m in queue.of("segment"))
    assert queue.of("end")[0]["exit_code"] == 0


def test_recording_failure_is_reported_not_raised(tmp_path):
    config, stt, mt = _providers()
    queue = Collector()
    runner = SessionRunner(run_id="r3", queue=queue)

    runner.start_recording(settings=config, stt=stt, translator=mt, diarizer=None,
                           source=tmp_path / "nope.wav",
                           work_root=tmp_path / "work", out=tmp_path / "x.md")
    runner.join(timeout=60)

    assert runner.status == "failed"
    logs = queue.of("log")
    assert logs, "a failure must produce a log message, not silence"
    assert "nope.wav" in " ".join(m["message"] for m in logs)
    assert queue.of("end")[0]["exit_code"] != 0


def test_segment_message_carries_every_field_the_page_renders():
    seg = Segment(chunk=Chunk(idx=3, start_s=1.5, end_s=9.5, wav_path="x.wav"),
                  lang="hi-IN", prob=0.91, text="नमस्ते", english="Hello",
                  status="ok", speaker="S1")
    assert SessionRunner.segment_message(1, seg, 0.08, True) == {
        "type": "segment", "seq": 1, "idx": 3, "start_s": 1.5, "end_s": 9.5,
        "lang": "hi-IN", "prob": 0.91, "text": "नमस्ते", "english": "Hello",
        "status": "ok", "speaker": "S1", "kept": True, "cost_inr": 0.08,
    }


def test_english_is_none_for_silence():
    seg = Segment(chunk=Chunk(idx=0, start_s=0.0, end_s=4.0, wav_path="x.wav"),
                  lang="unknown", prob=0.0, text="", english=None,
                  status="no_speech")
    assert SessionRunner.segment_message(0, seg, 0.0, False)["english"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/ui/test_session.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'omnilingual.ui.session'`.

- [ ] **Step 3: Write `omnilingual/ui/session.py`**

```python
"""One run, one worker thread, one stream of event dicts.

This is the ONLY module that knows how a run executes. The HTTP layer above it
sees nothing but queued dicts, so replacing this in-process thread with a child
process later is a change to this one file.

Pipeline callbacks arrive on pipeline threads and must reach an asyncio queue
owned by the server's loop; that handoff is the entire reason this class exists.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime
from pathlib import Path

from omnilingual.cache import JsonCache
from omnilingual.pipeline import run_from_chunks, work_dir_for
from omnilingual.pipeline import run as batch_run
from omnilingual.pipeline.live import LiveCapture, LiveOptions, run_live
from omnilingual.render.markdown import render, render_english_only

log = logging.getLogger(__name__)


class SessionRunner:
    """Owns one run's thread, its event stream, and its state."""

    def __init__(self, *, run_id: str, queue,
                 loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.run_id = run_id
        self._queue = queue
        self._loop = loop
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status = "idle"
        self._mode = "live"
        self._started = datetime.now()
        self._out: Path | None = None
        self._session_dir: Path | None = None
        self._segments = 0
        self._dropped = 0
        self._cost = 0.0
        self._cost_cap = 0.0
        self._seq = 0
        self._exit_code = 0
        self._error: str | None = None
        self._recoverable = False

    # --- events ---------------------------------------------------------

    def _emit(self, message: dict) -> None:
        """Hand one event to the server. Callable from any thread."""
        if self._loop is None:
            self._queue.put_nowait(message)
        else:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, message)

    @staticmethod
    def segment_message(seq: int, seg, cost_inr: float, kept: bool) -> dict:
        """Exactly the spec's §7.1 `segment` shape — one transcript row."""
        return {
            "type": "segment",
            "seq": seq,
            "idx": seg.chunk.idx if seg.chunk.idx >= 0 else seq,
            "start_s": round(seg.chunk.start_s, 2),
            "end_s": round(seg.chunk.end_s, 2),
            "lang": seg.lang,
            "prob": round(seg.prob, 4),
            "text": seg.text,
            "english": seg.english,
            "status": seg.status,
            "speaker": seg.speaker,
            "kept": kept,
            "cost_inr": round(cost_inr, 4),
        }

    def _push_state(self) -> None:
        self._emit({"type": "state", **self.snapshot()})

    # --- lifecycle ------------------------------------------------------

    def start_live(self, *, settings, stt, translator, diarizer,
                   opts: LiveOptions, capture_factory=LiveCapture) -> None:
        self._mode = "live"
        self._out = opts.out
        self._cost_cap = opts.max_cost
        self._status = "running"
        self._thread = threading.Thread(
            target=self._live, daemon=True,
            args=(settings, stt, translator, diarizer, opts, capture_factory))
        self._thread.start()
        self._push_state()

    def start_recording(self, *, settings, stt, translator, diarizer,
                        source: Path, work_root: Path, out: Path,
                        english_only: bool = False) -> None:
        self._mode = "recording"
        self._out = out
        self._status = "running"
        self._thread = threading.Thread(
            target=self._recording, daemon=True,
            args=(settings, stt, translator, diarizer, source, work_root, out,
                  english_only))
        self._thread.start()
        self._push_state()

    def recover(self, *, session_dir: Path, settings, stt, translator,
                diarizer, out: Path, english_only: bool = False) -> None:
        self._mode = "recover"
        self._out = out
        self._session_dir = session_dir
        self._status = "running"
        self._thread = threading.Thread(
            target=self._recovery, daemon=True,
            args=(session_dir, settings, stt, translator, diarizer, out,
                  english_only))
        self._thread.start()
        self._push_state()

    def stop(self) -> None:
        """Ask the run to stop. Idempotent, and safe before a thread exists."""
        with self._lock:
            if self._status == "running":
                self._status = "stopping"
        self._stop.set()
        self._push_state()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # --- run bodies -----------------------------------------------------

    def _live(self, settings, stt, translator, diarizer, opts: LiveOptions,
              capture_factory) -> None:
        work_root = opts.work_root or (opts.out.parent / ".omnilingual")
        before = set(work_root.glob("live-*")) if work_root.is_dir() else set()

        def on_status(message: str) -> None:
            self._emit({"type": "status", "message": message})

        def on_segment(seg, keep: bool) -> None:
            self._on_segment(seg, keep, 0.0)

        try:
            code = run_live(opts, settings, stt, translator, diarizer=diarizer,
                            status=on_status, capture_factory=capture_factory,
                            on_segment=on_segment, stop_event=self._stop)
        except BaseException as exc:  # noqa: BLE001 - reported, never raised out
            self._fail(exc)
            return

        after = set(work_root.glob("live-*")) if work_root.is_dir() else set()
        fresh = sorted(after - before)
        if fresh:
            self._session_dir = fresh[-1]
        self._exit_code = code
        self._finish("done" if code == 0 else "failed")

    def _recording(self, settings, stt, translator, diarizer, source: Path,
                   work_root: Path, out: Path, english_only: bool) -> None:
        try:
            work = work_dir_for(source, work_root)
            out.parent.mkdir(parents=True, exist_ok=True)

            def on_progress(index: int, total: int, seg) -> None:
                self._on_segment(seg, True, 0.0)
                self._emit({"type": "log", "level": "info",
                            "message": f"[{index}/{total}] chunks transcribed"})

            transcript = batch_run(source, work, settings, stt, translator,
                                   JsonCache(work / "cache"), on_progress,
                                   diarizer=diarizer)
            self._write(transcript, out, english_only)
            self._cost = transcript.cost.inr_estimate
            self._exit_code = 2 if any(s.status != "ok"
                                       for s in transcript.segments) else 0
        except BaseException as exc:  # noqa: BLE001
            self._fail(exc)
            return
        self._finish("done")

    def _recovery(self, session_dir: Path, settings, stt, translator, diarizer,
                  out: Path, english_only: bool) -> None:
        try:
            out.parent.mkdir(parents=True, exist_ok=True)

            def on_progress(index: int, total: int, seg) -> None:
                self._on_segment(seg, True, 0.0)
                self._emit({"type": "log", "level": "info",
                            "message": f"[{index}/{total}] chunks recovered"})

            transcript = run_from_chunks(session_dir, settings, stt, translator,
                                         JsonCache(session_dir / "cache"),
                                         on_progress, diarizer=diarizer)
            self._write(transcript, out, english_only)
            self._exit_code = 0
        except BaseException as exc:  # noqa: BLE001
            self._fail(exc)
            return
        self._finish("done")

    def _write(self, transcript, out: Path, english_only: bool) -> None:
        """The file is the artefact, so it is produced by the same renderer the
        CLI uses — the UI renders nothing itself."""
        out.write_text(render(transcript), encoding="utf-8")
        if english_only:
            out.with_suffix(".en.md").write_text(
                render_english_only(transcript), encoding="utf-8")

    # --- shared plumbing -------------------------------------------------

    def _on_segment(self, seg, keep: bool, delta: float) -> None:
        with self._lock:
            seq = self._seq
            self._seq += 1
            self._segments += 1
            if keep:
                self._cost += delta
            else:
                self._dropped += 1
        self._emit(self.segment_message(seq, seg, delta, keep))

    def _fail(self, exc: BaseException) -> None:
        log.exception("session %s failed", self.run_id, exc_info=exc)
        self._error = f"{type(exc).__name__}: {exc}"
        self._exit_code = 1
        self._emit({"type": "log", "level": "error", "message": self._error})
        self._finish("failed")

    def _finish(self, status: str) -> None:
        with self._lock:
            self._status = status
            self._recoverable = self._session_dir is not None
        self._push_state()
        self._emit({
            "type": "end",
            "exit_code": self._exit_code,
            "out": str(self._out) if self._out else None,
            "session_dir": (str(self._session_dir) if self._session_dir
                            else None),
            "recoverable": self._recoverable,
        })

    # --- read-only views -------------------------------------------------

    @property
    def status(self) -> str:
        return self._status

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    @property
    def out_path(self) -> Path | None:
        return self._out

    @property
    def recoverable(self) -> bool:
        return self._recoverable

    def snapshot(self) -> dict:
        return {
            "run_id": self.run_id,
            "mode": self._mode,
            "status": self._status,
            "elapsed_s": round((datetime.now() - self._started).total_seconds(), 1),
            "cost_inr": round(self._cost, 2),
            "cost_cap": self._cost_cap,
            "segments": self._segments,
            "dropped": self._dropped,
            "out": str(self._out) if self._out else None,
            "session_dir": (str(self._session_dir) if self._session_dir
                            else None),
            "error": self._error,
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/ui/test_session.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS. Nothing outside `tests/ui/` changed behaviour.

- [ ] **Step 6: Commit**

```bash
git add omnilingual/ui/session.py tests/ui/test_session.py
git commit -m "feat(ui): add SessionRunner for live, recording, and recovery runs"
```

---

### Task 7: The ASGI server — routes, WebSocket, and loopback hardening

**Files:**
- Create: `omnilingual/ui/server.py`
- Test: `tests/ui/test_server.py`

**Interfaces:**
- Consumes: `SessionRunner` (Task 6), `settings` / `secrets` / `audio` (Tasks 4-5); `load_settings`, `validate_chunk_bounds`, `validate_target_s`, `ConfigError`, `STT_PROVIDERS`, `MT_PROVIDERS`, `DEFAULT_STT_MODELS`, `DEFAULT_MT_MODELS` (`omnilingual.config`); `build_stt` (`omnilingual.stt`), `build_translator` (`omnilingual.translate`), `build_diarizer` (`omnilingual.diarize`); `LiveOptions` (`omnilingual.pipeline.live`); `ensure_ffmpeg`, `FfmpegMissingError` (`omnilingual.audio.normalize`); `CaptureError` (`omnilingual.audio.live_capture`).
- Produces:
  ```python
  def mint_token() -> str
  @dataclass
  class AppState:
      token: str
      port: int
      runs: dict[str, SessionRunner]
      queue: asyncio.Queue
      active: str | None
      loop: asyncio.AbstractEventLoop | None
  def create_app(*, token: str | None = None, port: int | None = None) -> FastAPI
  def _build(body: dict) -> dict     # validate + build providers, or raise
  ```
  Routes exactly as the spec's §7 table. Errors are JSON bodies `{"error": str}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/ui/test_server.py`:

```python
import pytest
from fastapi.testclient import TestClient

from omnilingual.ui.server import create_app, mint_token

HOST = "127.0.0.1:5599"


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("OMNILINGUAL_ENV_FILE", str(tmp_path / ".env"))
    token = mint_token()
    return TestClient(create_app(token=token, port=5599)), token


def _ok(token, **extra):
    return {"X-Omnilingual-Token": token, "Host": HOST, **extra}


def _start_body(**over):
    body = {"mode": "live", "stt": "sarvam", "mt": "mayura"}
    body.update(over)
    return body


def test_health_needs_no_token_and_reports_the_environment(api):
    client, _ = api
    res = client.get("/api/health", headers={"Host": HOST})
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert set(body) >= {"ok", "version", "arch", "macos", "arm64", "python"}


def test_request_without_a_token_is_forbidden(api):
    client, _ = api
    assert client.get("/api/defaults", headers={"Host": HOST}).status_code == 403


def test_request_with_a_wrong_token_is_forbidden(api):
    client, _ = api
    res = client.get("/api/defaults", headers=_ok("not-the-token"))
    assert res.status_code == 403


def test_foreign_host_header_is_rejected(api):
    client, token = api
    res = client.get("/api/defaults",
                     headers={"X-Omnilingual-Token": token, "Host": "evil.example"})
    assert res.status_code == 403


def test_foreign_origin_is_rejected(api):
    client, token = api
    res = client.get("/api/defaults",
                     headers=_ok(token, Origin="https://evil.example"))
    assert res.status_code == 403


def test_localhost_host_header_is_allowed(api):
    client, token = api
    res = client.get("/api/defaults",
                     headers={"X-Omnilingual-Token": token,
                              "Host": f"localhost:5599"})
    assert res.status_code == 200


def test_defaults_include_providers_and_models(api):
    client, token = api
    body = client.get("/api/defaults", headers=_ok(token)).json()
    assert "sarvam" in body["stt_providers"]
    assert "mayura" in body["mt_providers"]
    assert body["default_stt_models"]["sarvam"] == "saaras:v4"
    assert body["device"] == "Omnilingual"
    assert body["max_chunk_s"] == 28.0


def test_settings_round_trip(api):
    client, token = api
    res = client.put("/api/settings", headers=_ok(token),
                     json={"out": "/tmp/x.md", "stt": "groq"})
    assert res.status_code == 200
    assert client.get("/api/defaults",
                      headers=_ok(token)).json()["out"] == "/tmp/x.md"


def test_settings_reject_unknown_keys(api):
    client, token = api
    res = client.put("/api/settings", headers=_ok(token), json={"bogus": 1})
    assert res.status_code == 400
    assert "bogus" in res.json()["error"]


def test_keys_endpoint_returns_booleans_only(api, tmp_path):
    (tmp_path / ".env").write_text("GROQ_API_KEY=super-secret-value\n",
                                   encoding="utf-8")
    client, token = api
    res = client.get("/api/keys", headers=_ok(token))
    assert res.json() == {"SARVAM_API_KEY": False, "GROQ_API_KEY": True,
                          "GEMINI_API_KEY": False}
    assert "super-secret-value" not in res.text


def test_put_keys_writes_and_clear_removes(api, tmp_path):
    client, token = api
    assert client.put("/api/keys", headers=_ok(token),
                      json={"GROQ_API_KEY": "written-by-test"}).status_code == 200
    assert "written-by-test" in (tmp_path / ".env").read_text(encoding="utf-8")
    assert client.put("/api/keys", headers=_ok(token),
                      json={"GROQ_API_KEY": None}).status_code == 200
    assert "GROQ_API_KEY" not in (tmp_path / ".env").read_text(encoding="utf-8")


def test_put_keys_rejects_an_unknown_name(api):
    client, token = api
    res = client.put("/api/keys", headers=_ok(token),
                     json={"AWS_SECRET_ACCESS_KEY": "x"})
    assert res.status_code == 400


def test_start_rejects_an_unknown_stt_provider(api):
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(stt="not-a-provider"))
    assert res.status_code == 400
    assert "not-a-provider" in res.json()["error"]


def test_start_rejects_a_missing_api_key(api):
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body())
    assert res.status_code == 400
    assert "SARVAM_API_KEY" in res.json()["error"]


def test_start_rejects_bad_chunk_bounds(api):
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(min_chunk_s=5.0, max_chunk_s=40.0))
    assert res.status_code == 400
    assert "--max-chunk-s must be < 30" in res.json()["error"]


def test_start_rejects_a_target_outside_the_bounds(api):
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(target_s=99.0))
    assert res.status_code == 400
    assert "--target-s" in res.json()["error"]


def test_start_requires_a_source_in_recording_mode(api):
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(mode="recording", source=""))
    assert res.status_code == 400
    assert "source" in res.json()["error"].lower()


def test_start_rejects_a_missing_source_file(api, tmp_path):
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(mode="recording",
                                       source=str(tmp_path / "gone.wav")))
    assert res.status_code == 400
    assert "not found" in res.json()["error"]


def test_audio_endpoint_returns_the_readiness_shape(api, monkeypatch):
    from omnilingual.ui import audio
    monkeypatch.setattr(audio, "probe", lambda *a, **k: audio.Readiness(
        device=True, blackhole=True, ffmpeg=True, ffprobe=True,
        mic_authorized=True, output="Speakers", detail=[]))
    client, token = api
    body = client.get("/api/audio", headers=_ok(token)).json()
    assert body["ok"] is True
    assert body["device"] == "Omnilingual"
    assert body["mic_authorized"] is True


def test_websocket_requires_the_token(api):
    client, _ = api
    with pytest.raises(Exception):
        with client.websocket_connect("/ws", headers={"Host": HOST}):
            pass


def test_websocket_sends_state_on_connect(api):
    client, token = api
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        message = socket.receive_json()
    assert message["type"] == "state"
    assert message["status"] in {"idle", "done", "failed"}


def test_runs_endpoint_lists_nothing_before_a_run(api):
    client, token = api
    assert client.get("/api/runs", headers=_ok(token)).json()["runs"] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/ui/test_server.py -q`
Expected: FAIL — `ImportError: cannot import name 'create_app' from 'omnilingual.ui.server'`.

- [ ] **Step 3: Write `omnilingual/ui/server.py`**

```python
"""The ASGI app: routes, the WebSocket event stream, and loopback hardening.

Deliberately knows nothing about transcription. It validates a request, calls the
existing config and provider factories, hands off to a SessionRunner, and relays
queued events to the page. Every request is validated before a run starts, so a
bad panel produces a 400 rather than a session that dies halfway.
"""

from __future__ import annotations

import asyncio
import platform
import secrets as pysecrets
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from omnilingual.audio.live_capture import CaptureError
from omnilingual.audio.normalize import FfmpegMissingError, ensure_ffmpeg
from omnilingual.config import (
    DEFAULT_MT_MODELS,
    DEFAULT_STT_MODELS,
    MT_PROVIDERS,
    STT_PROVIDERS,
    ConfigError,
    load_settings,
    validate_chunk_bounds,
    validate_target_s,
)
from omnilingual.diarize import build_diarizer
from omnilingual.pipeline.live import LiveOptions
from omnilingual.stt import build_stt
from omnilingual.translate import build_translator

from omnilingual.ui import audio, secrets, settings
from omnilingual.ui.session import SessionRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
UI_VERSION = "1"
TOKEN_HEADER = "X-Omnilingual-Token"
START_ERRORS = (ConfigError, FfmpegMissingError, CaptureError, ValueError)


class _Denied(Exception):
    """Carries a prepared response out of the guard."""

    def __init__(self, response: JSONResponse) -> None:
        self.response = response


def mint_token() -> str:
    """A fresh, unguessable token per launch."""
    return pysecrets.token_urlsafe(32)


def _require_key(config) -> None:
    """Mirror the CLI's _require_keys: demand only what the choice needs.

    Picking free backends on both axes must not demand a Sarvam key, so each
    provider's credential is required by its own path.
    """
    if config.stt_provider == "sarvam" or config.mt_provider == "mayura":
        config.require_key()
    if config.stt_provider == "groq":
        config.require_groq_key()
    if config.mt_provider == "gemini":
        config.require_gemini_key()


@dataclass
class AppState:
    token: str
    port: int
    runs: dict[str, SessionRunner] = field(default_factory=dict)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    active: str | None = None
    loop: asyncio.AbstractEventLoop | None = None

    def free_run_id(self) -> str:
        return f"run-{len(self.runs) + 1}-{pysecrets.token_hex(4)}"


def _idle_state() -> dict:
    return {"run_id": None, "mode": "live", "status": "idle", "elapsed_s": 0.0,
            "cost_inr": 0.0, "cost_cap": 0.0, "segments": 0, "dropped": 0,
            "out": None, "session_dir": None, "error": None}


def _as_float(body: dict, name: str, fallback: float) -> float:
    try:
        return float(body.get(name, fallback))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


def _build(body: dict) -> dict:
    """Validate a request and construct the providers.

    Everything that can be wrong surfaces here, so a bad panel gives a 400 rather
    than a session that fails halfway. Validation is the CLI's own:
    load_settings rejects unknown providers and bad speaker counts, and the chunk
    bounds come from the shared validators rather than a second copy of the rule.
    """
    mode = body.get("mode", "live")
    if mode not in ("live", "recording"):
        raise ValueError(f"mode must be 'live' or 'recording', got {mode!r}")

    min_chunk_s = _as_float(body, "min_chunk_s", 5.0)
    max_chunk_s = _as_float(body, "max_chunk_s", 28.0)
    validate_chunk_bounds(min_chunk_s, max_chunk_s)
    if mode == "live":
        validate_target_s(_as_float(body, "target_s", 8.0),
                          min_chunk_s, max_chunk_s)

    langs = body.get("langs") or []
    if isinstance(langs, str):
        langs = [part.strip() for part in langs.split(",") if part.strip()]

    diarize = bool(body.get("diarize"))
    config = load_settings(
        langs=list(langs),
        stt_provider=body.get("stt") or "sarvam",
        stt_model=body.get("stt_model") or None,
        mt_provider=body.get("mt") or "mayura",
        mt_model=body.get("mt_model") or None,
        diarizer=body.get("diarizer") or ("sherpa" if diarize else None),
        num_speakers=(int(body["num_speakers"])
                      if diarize and body.get("num_speakers") else None),
        max_chunk_s=max_chunk_s,
        min_chunk_s=min_chunk_s,
    )
    providers = {
        "settings": config,
        "stt": build_stt(config),
        "translator": build_translator(config),
        "diarizer": build_diarizer(config) if config.diarizer else None,
    }
    _require_key(config)

    out = Path(body.get("out") or "standup.md").expanduser()
    work_dir = body.get("work_dir") or ""
    work_root = (Path(work_dir).expanduser() if work_dir
                 else out.parent / ".omnilingual")
    english_only = bool(body.get("english_only"))
    ensure_ffmpeg()

    if mode == "live":
        kwargs = dict(
            providers,
            opts=LiveOptions(
                out=out,
                device=body.get("device") or "Omnilingual",
                mic_only=bool(body.get("mic_only")),
                target_s=_as_float(body, "target_s", 8.0),
                max_chunk_s=max_chunk_s,
                min_chunk_s=min_chunk_s,
                noise_db=_as_float(body, "noise_db", -35.0),
                stt_workers=int(body.get("stt_workers") or 2),
                max_cost=_as_float(body, "max_cost", 50.0),
                work_root=work_root,
                english_only=english_only,
            ),
        )
    else:
        source = body.get("source") or ""
        if not source:
            raise ValueError("source is required in recording mode")
        source_path = Path(source).expanduser()
        if not source_path.is_file():
            raise ValueError(f"source not found: {source_path}")
        kwargs = dict(providers, source=source_path, work_root=work_root,
                      out=out, english_only=english_only)

    return {"mode": mode, "kwargs": kwargs}


def create_app(*, token: str | None = None, port: int | None = None) -> FastAPI:
    token = token or mint_token()
    state = AppState(token=token, port=port or 0)
    app = FastAPI(title="omnilingual-ui", docs_url=None, redoc_url=None)
    app.state.ui = state

    def host_ok(host: str) -> bool:
        """DNS-rebinding defence: only the two loopback spellings are legal."""
        return host in (f"127.0.0.1:{state.port}", f"localhost:{state.port}")

    def guard(request: Request) -> None:
        if not host_ok(request.headers.get("host", "")):
            raise _Denied(JSONResponse({"error": "bad host"}, status_code=403))
        origin = request.headers.get("origin")
        if origin and origin not in (f"http://127.0.0.1:{state.port}",
                                     f"http://localhost:{state.port}"):
            raise _Denied(JSONResponse({"error": "bad origin"}, status_code=403))
        if request.headers.get(TOKEN_HEADER) != state.token:
            raise _Denied(JSONResponse({"error": "bad token"}, status_code=403))

    @app.exception_handler(_Denied)
    async def _denied_handler(request: Request, exc: _Denied) -> JSONResponse:
        return exc.response

    @app.get("/api/health")
    async def health() -> dict:
        # Unguarded on purpose: the window must be able to prove the server is up
        # before it has a token. It exposes no user data.
        return {"ok": True, "version": UI_VERSION,
                "arch": platform.machine(), "macos": platform.mac_ver()[0],
                "arm64": platform.machine() == "arm64",
                "python": sys.version.split()[0]}

    @app.get("/api/defaults")
    async def defaults(request: Request) -> dict:
        guard(request)
        return {**settings.load(),
                "stt_providers": list(STT_PROVIDERS),
                "mt_providers": list(MT_PROVIDERS),
                "default_stt_models": dict(DEFAULT_STT_MODELS),
                "default_mt_models": dict(DEFAULT_MT_MODELS),
                "last_output_dir": str(REPO_ROOT)}

    @app.put("/api/settings")
    async def put_settings(request: Request):
        guard(request)
        try:
            return settings.save(await request.json())
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/keys")
    async def get_keys(request: Request) -> dict:
        guard(request)
        return secrets.present()

    @app.put("/api/keys")
    async def put_keys(request: Request):
        guard(request)
        try:
            for name, value in (await request.json()).items():
                if value is None:
                    secrets.clear(name)
                else:
                    secrets.set_key(name, str(value))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return secrets.present()

    @app.get("/api/audio")
    async def get_audio(request: Request) -> dict:
        guard(request)
        return audio.probe().as_dict()

    @app.post("/api/audio/setup")
    async def post_audio_setup(request: Request):
        guard(request)
        try:
            for line in audio.setup():
                state.queue.put_nowait({"type": "log", "level": "info",
                                        "message": line})
        except Exception as exc:  # noqa: BLE001 - reported to the page
            return JSONResponse({"error": str(exc)}, status_code=500)
        return audio.probe(mic=False).as_dict()

    @app.post("/api/audio/restart-daemon")
    async def post_restart_daemon(request: Request):
        guard(request)
        try:
            audio.restart_daemon()
        except (OSError, subprocess.SubprocessError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        return audio.probe(mic=False).as_dict()

    @app.post("/api/session/start")
    async def post_start(request: Request):
        guard(request)
        if state.active is not None:
            return JSONResponse({"error": "a run is already in progress; "
                                           "stop it first"}, status_code=409)
        try:
            built = _build(await request.json())
        except START_ERRORS as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        runner = SessionRunner(run_id=state.free_run_id(), queue=state.queue,
                               loop=state.loop)
        state.runs[runner.run_id] = runner
        state.active = runner.run_id
        if built["mode"] == "live":
            runner.start_live(**built["kwargs"])
        else:
            runner.start_recording(**built["kwargs"])
        return {"run_id": runner.run_id, "mode": built["mode"]}

    @app.post("/api/session/stop")
    async def post_stop(request: Request):
        guard(request)
        runner = state.runs.get(state.active or "")
        if runner is None:
            return JSONResponse({"error": "no run is active"}, status_code=404)
        runner.stop()
        return {"run_id": runner.run_id, "status": runner.status}

    @app.get("/api/runs")
    async def get_runs(request: Request) -> dict:
        guard(request)
        return {"runs": [
            {"run_id": r.run_id, "status": r.status,
             "out": str(r.out_path) if r.out_path else None,
             "session_dir": str(r.session_dir) if r.session_dir else None,
             "recoverable": r.recoverable}
            for r in state.runs.values()]}

    @app.post("/api/session/recover")
    async def post_recover(request: Request):
        guard(request)
        if state.active is not None:
            return JSONResponse({"error": "a run is already in progress; "
                                           "stop it first"}, status_code=409)
        body = await request.json()
        session_dir = Path(body.get("session_dir") or "")
        if not session_dir.is_dir():
            return JSONResponse(
                {"error": f"{session_dir} is not a session dir"}, status_code=400)
        request_body = {**settings.load(), "mode": "recording",
                        "out": body.get("out") or "recovered.md"}
        try:
            built = _build(request_body)
        except START_ERRORS as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        runner = SessionRunner(run_id=state.free_run_id(), queue=state.queue,
                               loop=state.loop)
        state.runs[runner.run_id] = runner
        state.active = runner.run_id
        kwargs = built["kwargs"]
        runner.recover(session_dir=session_dir, out=kwargs["out"],
                       **{k: kwargs[k] for k in
                          ("settings", "stt", "translator", "diarizer")})
        return {"run_id": runner.run_id, "mode": "recover"}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        # The browser WebSocket API cannot set headers on the handshake, so the
        # token may arrive as a query parameter as well as a header.
        supplied = (socket.headers.get(TOKEN_HEADER)
                    or socket.query_params.get("token"))
        if not host_ok(socket.headers.get("host", "")) or supplied != state.token:
            await socket.close(code=1008)
            return
        await socket.accept()
        try:
            runner = state.runs.get(state.active or "")
            await socket.send_json(
                {"type": "state", **(runner.snapshot() if runner
                                    else _idle_state())})
            while True:
                message = await state.queue.get()
                await socket.send_json(message)
                if message["type"] == "end":
                    state.active = None
        except (WebSocketDisconnect, RuntimeError):
            # A disconnect must not kill the run: it lives in the server, and a
            # reconnecting page re-reads state from this same registry.
            return

    @app.get("/token.js", response_class=PlainTextResponse)
    async def token_js() -> str:
        # The page needs the token but must not have it hard-coded in the HTML.
        # Same-origin and read-only, so it needs no token of its own.
        return f"window.OMNILINGUAL_TOKEN = {state.token!r};"

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)),
                  name="static")

    return app
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/ui/test_server.py -q`
Expected: PASS (22 tests).

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add omnilingual/ui/server.py tests/ui/test_server.py
git commit -m "feat(ui): add ASGI server with routes, event stream, and loopback guards"
```

---

### Task 8: The single page

One page, two modes. `app.js` is a thin renderer over the server's messages — it holds no transcription logic and does no transcript formatting of its own.

**Files:**
- Create: `omnilingual/ui/static/index.html`
- Create: `omnilingual/ui/static/style.css`
- Create: `omnilingual/ui/static/app.js`
- Test: `tests/ui/test_static.py`

**Interfaces:**
- Consumes: `GET /token.js`, `GET /api/defaults`, `GET /api/keys`, `PUT /api/keys`, `GET /api/audio`, `POST /api/session/start`, `POST /api/session/stop`, `POST /api/audio/setup`, `POST /api/audio/restart-daemon`, `WS /ws` (all from Task 7).
- Produces: no Python API. The page reads `window.OMNILINGUAL_TOKEN` and sends it as `X-Omnilingual-Token`, and passes it as `?token=` on the WebSocket.

- [ ] **Step 1: Write the failing test**

Create `tests/ui/test_static.py`:

```python
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "omnilingual" / "ui" / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_all_three_assets_exist():
    for name in ("index.html", "style.css", "app.js"):
        assert (STATIC / name).is_file(), f"{name} missing"


def test_page_fetches_the_token_and_sends_it_on_every_call():
    js = _read("app.js")
    assert "/token.js" in js, "the token must be fetched, not hard-coded"
    assert "X-Omnilingual-Token" in js


def test_page_opens_a_websocket_with_the_token():
    js = _read("app.js")
    assert re.search(r"/ws\b", js)
    assert "WebSocket" in js
    assert "token=" in js


def test_page_covers_both_modes_and_every_parameter_control():
    html = _read("index.html")
    for control in ("mode", "source", "out", "stt", "stt_model", "mt", "mt_model",
                    "diarize", "num_speakers", "langs", "english_only", "work_dir",
                    "device", "mic_only", "target_s", "max_chunk_s", "min_chunk_s",
                    "noise_db", "stt_workers", "max_cost"):
        assert f'id="{control}"' in html, f"panel control {control} missing"
    assert "Set up audio" in html
    assert "Restart audio daemon" in html


def test_page_renders_every_server_message_type():
    js = _read("app.js")
    for kind in ("state", "segment", "status", "log", "end"):
        assert f'"{kind}"' in js, f"no handler for the {kind} message"


def test_page_shows_key_presence_not_key_values():
    # GET /api/keys returns booleans only, so the page must render presence.
    js = _read("app.js")
    assert "/api/keys" in js
    assert "configured" in js


def test_page_offers_recovery_only_when_the_run_is_recoverable():
    assert "recoverable" in _read("app.js")


def test_page_never_switches_the_system_output_itself():
    # Routing output silently breaks volume keys; only the server may report it.
    assert "SwitchAudioSource" not in _read("app.js")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/ui/test_static.py -q`
Expected: FAIL — `AssertionError: index.html missing`.

- [ ] **Step 3: Write `omnilingual/ui/static/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>omnilingual</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<header>
  <h1>omnilingual</h1>
  <div id="banner" class="banner hidden"></div>
</header>

<main>
  <section id="panel">
    <div class="modes">
      <label><input type="radio" name="mode" value="live" checked> Live</label>
      <label><input type="radio" name="mode" value="recording"> Recording</label>
    </div>

    <label>Recording file
      <input id="source" type="text" placeholder="/path/to/meeting.m4a"></label>
    <label>Output file
      <input id="out" type="text" value="standup.md"></label>

    <label>Speech-to-text <select id="stt"></select></label>
    <label>STT model
      <input id="stt_model" type="text" placeholder="provider default"></label>

    <label>Translate <select id="mt"></select></label>
    <label>MT model
      <input id="mt_model" type="text" placeholder="provider default"></label>

    <label><input id="diarize" type="checkbox"> Label speakers</label>
    <label>Speakers <input id="num_speakers" type="number" min="2" value="3"></label>
    <label>Languages
      <input id="langs" type="text" placeholder="blank = auto-detect"></label>
    <label><input id="english_only" type="checkbox">
      Also write an English-only file</label>
    <label>Work directory
      <input id="work_dir" type="text" placeholder=".omnilingual"></label>

    <fieldset id="live-only">
      <legend>Live capture</legend>
      <label>Device <input id="device" type="text" value="Omnilingual"></label>
      <label><input id="mic_only" type="checkbox"> Microphone only</label>
      <label>Target seconds
        <input id="target_s" type="number" step="0.5" value="8"></label>
      <label>Max chunk s
        <input id="max_chunk_s" type="number" step="0.5" value="28"></label>
      <label>Min chunk s
        <input id="min_chunk_s" type="number" step="0.5" value="5"></label>
      <label>Noise floor dBFS
        <input id="noise_db" type="number" step="1" value="-35"></label>
      <label>STT workers
        <input id="stt_workers" type="number" min="1" value="2"></label>
      <label>Cost cap Rs
        <input id="max_cost" type="number" step="1" value="50"></label>
    </fieldset>

    <fieldset>
      <legend>API keys</legend>
      <div id="keys"></div>
    </fieldset>

    <div class="actions">
      <button id="start" type="button">Start</button>
      <button id="stop" type="button" disabled>Stop</button>
      <button id="setup-audio" type="button">Set up audio</button>
      <button id="restart-daemon" type="button">Restart audio daemon</button>
    </div>
  </section>

  <section id="output">
    <div id="summary" class="summary"></div>
    <div id="log" class="log"></div>
    <table id="rows">
      <thead>
        <tr><th>Time</th><th>Lang</th><th>Speaker</th>
            <th>Original</th><th>English</th></tr>
      </thead>
      <tbody id="rows-body"></tbody>
    </table>
    <div id="final" class="final hidden"></div>
  </section>
</main>
<script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Write `omnilingual/ui/static/style.css`**

```css
:root {
  --bg: #12141a;
  --panel: #1b1e26;
  --line: #2b303c;
  --text: #e7e9ee;
  --muted: #9aa3b2;
  --accent: #7aa2f7;
  --warn: #e0af68;
  --error: #f7768e;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.5 -apple-system, "SF Pro Text", system-ui, sans-serif;
}
header { padding: 12px 20px; border-bottom: 1px solid var(--line); }
h1 { margin: 0; font-size: 16px; letter-spacing: .02em; }
main { display: grid; grid-template-columns: 340px 1fr; min-height: calc(100vh - 46px); }
#panel { padding: 16px 20px; border-right: 1px solid var(--line); background: var(--panel); }
#panel label { display: block; margin-bottom: 10px; color: var(--muted); font-size: 12px; }
#panel input[type=text], #panel input[type=number], #panel select {
  width: 100%; margin-top: 4px; padding: 6px 8px;
  background: var(--bg); color: var(--text);
  border: 1px solid var(--line); border-radius: 6px;
}
fieldset { border: 1px solid var(--line); border-radius: 8px; margin: 0 0 14px; padding: 10px 12px; }
legend { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
.modes { display: flex; gap: 16px; margin-bottom: 12px; }
.modes label, .key-row label, #panel label.check { display: flex; align-items: center; gap: 6px; }
.actions { display: flex; flex-wrap: wrap; gap: 8px; }
button {
  padding: 7px 14px; border-radius: 6px; cursor: pointer;
  border: 1px solid var(--line); background: var(--bg); color: var(--text);
}
button:disabled { opacity: .45; cursor: default; }
#start { border-color: var(--accent); color: var(--accent); }
#output { padding: 16px 20px; overflow-y: auto; }
.summary { color: var(--muted); margin-bottom: 8px; }
.log {
  max-height: 160px; overflow-y: auto;
  font-family: ui-monospace, Menlo, monospace; font-size: 11px;
  color: var(--muted); white-space: pre-wrap;
}
.log .error { color: var(--error); }
.log .warn { color: var(--warn); }
table { width: 100%; border-collapse: collapse; margin-top: 12px; }
th {
  text-align: left; font-size: 11px; text-transform: uppercase;
  letter-spacing: .06em; color: var(--muted);
  border-bottom: 1px solid var(--line); padding: 6px 8px;
}
td { padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
tr.dropped { opacity: .45; }
td.english { color: var(--accent); }
td.speaker { color: var(--warn); font-weight: 600; }
.banner {
  margin-top: 10px; padding: 8px 12px; border-radius: 6px;
  border: 1px solid var(--warn); color: var(--warn);
}
.final { margin-top: 16px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px; }
.key-row { display: flex; align-items: center; gap: 6px; margin-bottom: 8px; }
.key-row input { flex: 1; margin: 0 !important; }
.hidden { display: none; }
```

- [ ] **Step 5: Write `omnilingual/ui/static/app.js`**

```javascript
// A thin renderer over the server's messages. No transcription logic and no
// formatting of the transcript itself: the saved Markdown file is the artefact,
// and this only displays what the pipeline produced.
const TOKEN_HEADER = "X-Omnilingual-Token";
let token = null;
let socket = null;

const $ = (id) => document.getElementById(id);

async function loadToken() {
  // Fetched at load rather than inlined, so it never sits in the HTML.
  await fetch("/token.js");
  token = window.OMNILINGUAL_TOKEN;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      [TOKEN_HEADER]: token,
      ...(options.headers || {}),
    },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `request failed (${res.status})`);
  return body;
}

function showBanner(message, level = "warn") {
  const banner = $("banner");
  banner.textContent = message;
  banner.classList.remove("hidden");
  banner.style.borderColor = level === "error" ? "var(--error)" : "var(--warn)";
}

function clearBanner() {
  $("banner").classList.add("hidden");
}

function fillSelect(id, values, current) {
  const select = $(id);
  select.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = value === current;
    select.appendChild(option);
  }
}

async function loadDefaults() {
  const d = await api("/api/defaults");
  fillSelect("stt", d.stt_providers, d.stt);
  fillSelect("mt", d.mt_providers, d.mt);
  $("stt_model").placeholder = d.default_stt_models[d.stt] || "provider default";
  $("mt_model").placeholder = d.default_mt_models[d.mt] || "provider default";
  for (const key of ["out", "source", "device", "work_dir", "langs"]) {
    if (d[key]) $(key).value = d[key];
  }
  for (const key of ["target_s", "max_chunk_s", "min_chunk_s", "noise_db",
                     "stt_workers", "max_cost", "num_speakers"]) {
    if (d[key] !== undefined && d[key] !== null) $(key).value = d[key];
  }
  for (const key of ["diarize", "mic_only", "english_only"]) {
    if (d[key] !== undefined) $(key).checked = Boolean(d[key]);
  }
  const radio = document.querySelector(`input[name=mode][value="${d.mode}"]`);
  if (radio) radio.checked = true;
  toggleMode();
}

async function loadKeys() {
  const present = await api("/api/keys");
  const box = $("keys");
  box.innerHTML = "";
  for (const [name, isSet] of Object.entries(present)) {
    const row = document.createElement("div");
    row.className = "key-row";
    const label = document.createElement("label");
    // Presence only: a stored value never leaves the server process.
    label.textContent = `${name}: ${isSet ? "configured" : "not configured"}`;
    const field = document.createElement("input");
    field.type = "password";
    field.placeholder = "new value";
    const save = document.createElement("button");
    save.textContent = "Save";
    save.onclick = async () => {
      if (!field.value) return;
      await api("/api/keys", {
        method: "PUT",
        body: JSON.stringify({ [name]: field.value }),
      });
      field.value = "";
      loadKeys();
    };
    const clearBtn = document.createElement("button");
    clearBtn.textContent = "Clear";
    clearBtn.onclick = async () => {
      await api("/api/keys", {
        method: "PUT",
        body: JSON.stringify({ [name]: null }),
      });
      loadKeys();
    };
    row.append(label, field, save, clearBtn);
    box.appendChild(row);
  }
}

async function checkAudio() {
  const ready = await api("/api/audio");
  if (ready.ok) {
    clearBanner();
    return;
  }
  showBanner(`Audio capture is not ready: ${ready.detail.join("; ")}`);
}

function currentMode() {
  return document.querySelector("input[name=mode]:checked").value;
}

function toggleMode() {
  const live = currentMode() === "live";
  $("live-only").classList.toggle("hidden", !live);
  $("source").disabled = live;
}

function collect() {
  const live = currentMode() === "live";
  const body = {
    mode: live ? "live" : "recording",
    source: $("source").value,
    out: $("out").value,
    stt: $("stt").value,
    stt_model: $("stt_model").value || null,
    mt: $("mt").value,
    mt_model: $("mt_model").value || null,
    diarize: $("diarize").checked,
    num_speakers: $("diarize").checked ? Number($("num_speakers").value) : null,
    langs: $("langs").value.split(",").map((s) => s.trim()).filter(Boolean),
    english_only: $("english_only").checked,
    work_dir: $("work_dir").value,
  };
  if (live) {
    Object.assign(body, {
      device: $("device").value,
      mic_only: $("mic_only").checked,
      target_s: Number($("target_s").value),
      max_chunk_s: Number($("max_chunk_s").value),
      min_chunk_s: Number($("min_chunk_s").value),
      noise_db: Number($("noise_db").value),
      stt_workers: Number($("stt_workers").value),
      max_cost: Number($("max_cost").value),
    });
  }
  return body;
}

function fmtTime(seconds) {
  const total = Math.floor(seconds || 0);
  const mm = String(Math.floor(total / 60)).padStart(2, "0");
  const ss = String(total % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

function addRow(message) {
  const tr = document.createElement("tr");
  if (!message.kept) tr.className = "dropped";
  for (const value of [fmtTime(message.start_s), message.lang || "",
                       message.speaker || ""]) {
    const td = document.createElement("td");
    if (value) td.textContent = value;
    if (value === message.speaker) td.className = "speaker";
    tr.appendChild(td);
  }
  const original = document.createElement("td");
  original.textContent = message.text || "";
  tr.appendChild(original);
  // The English cell stays empty when the source was already English, when the
  // segment failed, or for silence — the same rule the Markdown renderer uses.
  const english = document.createElement("td");
  english.className = "english";
  english.textContent = message.english || "";
  tr.appendChild(english);
  $("rows-body").appendChild(tr);
  tr.scrollIntoView({ block: "nearest" });
}

function log(message, level = "info") {
  const line = document.createElement("div");
  line.className = level;
  line.textContent = message;
  $("log").appendChild(line);
  $("log").scrollTop = $("log").scrollHeight;
}

function renderState(message) {
  $("summary").textContent =
    `${message.status} · ${fmtTime(message.elapsed_s)} · ` +
    `${message.segments} segments` +
    (message.dropped ? ` · ${message.dropped} dropped` : "") +
    (message.cost_cap ? ` · Rs ${message.cost_inr} of Rs ${message.cost_cap}` : "");
  const running = message.status === "running" || message.status === "stopping";
  $("stop").disabled = !running;
  $("start").disabled = running;
}

function renderEnd(message) {
  $("stop").disabled = true;
  $("start").disabled = false;
  const panel = $("final");
  panel.classList.remove("hidden");
  panel.innerHTML = "";
  const summary = document.createElement("div");
  summary.textContent = message.out ? `Saved ${message.out}`
                                     : "Run finished without an output file";
  panel.appendChild(summary);
  if (message.recoverable && message.session_dir) {
    const button = document.createElement("button");
    button.textContent = "Recover this session";
    button.onclick = () => recover(message.session_dir);
    panel.appendChild(button);
  }
}

function render(message) {
  switch (message.type) {
    case "state": renderState(message); break;
    case "segment": addRow(message); break;
    case "status": log(message.message); break;
    case "log": log(message.message, message.level); break;
    case "end": renderEnd(message); break;
    default: break;
  }
}

async function recover(sessionDir) {
  try {
    $("rows-body").innerHTML = "";
    await api("/api/session/recover", {
      method: "POST",
      body: JSON.stringify({ session_dir: sessionDir, out: $("out").value }),
    });
  } catch (err) {
    showBanner(err.message, "error");
  }
}

function connect() {
  socket = new WebSocket(
    `ws://${location.host}/ws?token=${encodeURIComponent(token)}`);
  socket.onmessage = (event) => render(JSON.parse(event.data));
  socket.onclose = () => log("event stream closed; the run continues", "warn");
}

async function start() {
  try {
    clearBanner();
    $("rows-body").innerHTML = "";
    $("log").innerHTML = "";
    $("final").classList.add("hidden");
    const body = collect();
    await api("/api/session/start", {
      method: "POST", body: JSON.stringify(body),
    });
    // Preferences are remembered so the panel reopens where the user worked.
    await api("/api/settings", { method: "PUT", body: JSON.stringify(body) });
  } catch (err) {
    showBanner(err.message, "error");
  }
}

async function stop() {
  try {
    await api("/api/session/stop", { method: "POST" });
  } catch (err) {
    showBanner(err.message, "error");
  }
}

async function post(path) {
  try {
    await api(path, { method: "POST" });
    await checkAudio();
  } catch (err) {
    showBanner(err.message, "error");
  }
}

async function main() {
  await loadToken();
  await loadDefaults();
  await loadKeys();
  await checkAudio();
  connect();
  $("start").onclick = start;
  $("stop").onclick = stop;
  $("setup-audio").onclick = () => post("/api/audio/setup");
  $("restart-daemon").onclick = () => post("/api/audio/restart-daemon");
  for (const radio of document.querySelectorAll("input[name=mode]")) {
    radio.onchange = toggleMode;
  }
  $("stt").onchange = loadDefaults;
  $("mt").onchange = loadDefaults;
}

main();
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/ui/test_static.py -q`
Expected: PASS (8 tests).

- [ ] **Step 7: Re-run the server tests**

The `/ws` token handling already accepts the query parameter from Task 7, so nothing there changes.

Run: `uv run pytest tests/ui/ -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add omnilingual/ui/static/ tests/ui/test_static.py
git commit -m "feat(ui): add the single-page panel, transcript table, and log strip"
```

---

### Task 9: The launcher

Picks a free loopback port, serves on a daemon thread, opens a native window, and falls back to the browser when `pywebview` is unavailable.

**Files:**
- Create: `omnilingual/ui/__main__.py`
- Test: `tests/ui/test_main.py`

**Interfaces:**
- Consumes: `create_app`, `mint_token` (Task 7).
- Produces:
  ```python
  def build_parser() -> argparse.ArgumentParser
  def free_port() -> int
  def serve(port: int, *, host: str = "127.0.0.1") -> tuple[object, threading.Thread]
  def open_window(url: str, *, no_window: bool = False) -> None
  def main(argv: list[str] | None = None) -> int
  ```

- [ ] **Step 1: Write the failing tests**

Create `tests/ui/test_main.py`:

```python
import socket

import pytest

from omnilingual.ui.__main__ import free_port, main


def test_free_port_returns_a_bindable_loopback_port():
    port = free_port()
    assert 1024 < port < 65536
    # The whole point is that nothing else holds it, so we must be able to bind.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", port))


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--no-window" in capsys.readouterr().out


def test_main_accepts_the_documented_flags():
    # Parsing must succeed without starting anything, so --help is the only
    # zero-side-effect invocation; here we only assert the parser knows the flags.
    import argparse
    from omnilingual.ui.__main__ import build_parser
    parsed = build_parser().parse_args(["--no-window", "--port", "1234",
                                        "--host", "127.0.0.1"])
    assert parsed.no_window is True
    assert parsed.port == 1234
    assert parsed.host == "127.0.0.1"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/ui/test_main.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'omnilingual.ui.__main__'`.

- [ ] **Step 3: Write `omnilingual/ui/__main__.py`**

```python
"""Launch the UI: serve on a free loopback port and open a window.

The window is a convenience, never a dependency. If pywebview is unavailable or
refuses to open, the app falls back to the system browser and everything still
works, because all the logic lives in the server.
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import threading
import time
import webbrowser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnilingual-ui",
        description="Run the omnilingual UI in a local window.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; loopback only by default")
    parser.add_argument("--port", type=int, default=0,
                        help="port to bind; 0 picks a free one")
    parser.add_argument("--no-window", action="store_true",
                        help="serve only and print the URL")
    return parser


def free_port() -> int:
    """Ask the OS for an unused loopback port, then release it.

    Binding to 0 is the only race-free way to do this; a fixed port would make
    two instances collide.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def serve(port: int, *, host: str = "127.0.0.1"):
    """Start uvicorn on a daemon thread. Returns (server, thread)."""
    import uvicorn

    from omnilingual.ui.server import create_app

    config = uvicorn.Config(create_app(port=port), host=host, port=port,
                            log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def open_window(url: str, *, no_window: bool = False) -> None:
    """Open a native window, falling back to the browser."""
    if no_window:
        return
    try:
        import webview
    except ImportError:
        webbrowser.open(url)
        return
    try:
        webview.create_window("omnilingual", url, width=1280, height=860)
        webview.start()
    except Exception:  # noqa: BLE001 - a window is never load-bearing
        webbrowser.open(url)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    port = args.port or free_port()
    server, thread = serve(port, host=args.host)
    url = f"http://{args.host}:{port}/"

    # uvicorn binds inside its own thread, so wait for the socket rather than
    # assuming the thread start means the port is live.
    for _ in range(100):
        try:
            with socket.create_connection((args.host, port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        print(f"server did not come up on {url}", file=sys.stderr)
        return 1

    print(f"omnilingual UI on {url}")
    open_window(url, no_window=args.no_window)

    server.should_exit = True
    thread.join(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Delete the unused `subprocess` import from the header — the launcher never spawns anything.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/ui/test_main.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Smoke-test the launcher end to end**

Run in the background: `uv run omnilingual-ui --no-window --port 8765 &`
Then, in another shell:
`curl -s http://127.0.0.1:8765/api/health`
Expected: JSON containing `"ok": true`.
Then confirm the page is served: `curl -s http://127.0.0.1:8765/ | head -3`
Expected: the `<!doctype html>` line and the `<title>omnilingual</title>` line.
Finally kill the process. This is the first proof that the three defences in Task 7 do not lock the real page out.

- [ ] **Step 6: Commit**

```bash
git add omnilingual/ui/__main__.py tests/ui/test_main.py
git commit -m "feat(ui): add the launcher with loopback port, server thread, and window"
```

---

### Task 10: Register the `ui` extra with the setup script

Required, not optional. `scripts/setup-mac.sh` converges extras with `uv sync --all-extras --no-extra <unwanted>`. Its `ALL_EXTRAS` constant is the subtraction list, so a `ui` extra that is not in `ALL_EXTRAS` would be installed unconditionally by `--all-extras` and could never be deselected — silently breaking the convergence property the script was built and reviewed for.

**Files:**
- Modify: `scripts/setup-mac.sh` (the `ALL_EXTRAS` constant, the capability prompt, and the usage comment)
- Test: manual verification via `--dry-run`

**Interfaces:**
- Consumes: nothing from earlier tasks except the `ui` extra name from Task 2.
- Produces: a setup script whose capability list includes `ui`, so `uv sync --all-extras --no-extra ui` can remove it.

- [ ] **Step 1: Confirm the bug exists**

Run: `grep -n 'ALL_EXTRAS=' scripts/setup-mac.sh`
Expected: a line listing the extras without `ui`. This is the defect: `--all-extras` would install `ui` unconditionally.

- [ ] **Step 2: Add `ui` to `ALL_EXTRAS`**

In `scripts/setup-mac.sh`, change:

```bash
ALL_EXTRAS="local-stt diarize local-mt"
```

to:

```bash
ALL_EXTRAS="local-stt diarize local-mt ui"
```

- [ ] **Step 3: Add `ui` to the capability prompt**

Find the capabilities prompt inside `prompt_extras` (the block that presents `local-stt`, `diarize`, and `local-mt` and sets `EXTRAS`) and add a matching entry, so the description and the `in_list` gate both name `ui`. The entry must read as a capability, for example: `ui — the desktop UI window (omnilingual-ui)`.

- [ ] **Step 4: Update the header comment**

The usage block at the top of the script lists what it installs. Mention the UI so the script's own documentation does not go stale.

- [ ] **Step 5: Verify convergence with a dry run**

Run: `bash scripts/setup-mac.sh --dry-run --yes`
Expected: the plan either omits the extras line (already converged) or shows `uv sync --all-extras --no-extra ui` when `ui` is not selected. It must never show a bare `uv sync --all-extras` with no `--no-extra ui`.

Run again with `ui` selected in the saved config, and confirm the line disappears. Nothing on disk should change under `--dry-run`.

- [ ] **Step 6: Verify the script still parses and lints as before**

Run: `bash -n scripts/setup-mac.sh`
Expected: no output, exit 0.

- [ ] **Step 7: Commit**

```bash
git add scripts/setup-mac.sh
git commit -m "build: teach setup-mac.sh about the ui extra so it can be deselected"
```

---

### Task 11: Full verification against the spec's acceptance criteria

Nothing new is written here. This task proves phase 1 meets the spec's §16 ship list, and is the gate before handing over to phase 2.

**Files:**
- No file changes expected. Fix anything this task surfaces, in the task that owns it.

**Interfaces:**
- Consumes: everything from Tasks 1-10.
- Produces: verification evidence for each of the seven phase 1 criteria.

- [ ] **Step 1: The whole suite is green**

Run: `uv run pytest -q`
Expected: PASS, with only the pre-existing opt-in real-model tests skipped. Record the counts.

- [ ] **Step 2: The CLI is unchanged**

Run: `uv run omnilingual transcribe --help` and `uv run omnilingual live --help`
Expected: identical flags to before this plan. Task 1 changed only *where* the chunk-bound rule lives, and Task 3 added two keyword-only params that default to `None`.

Run: `uv run pytest tests/test_cli.py tests/test_cli_live.py tests/test_cli_ask.py -q`
Expected: PASS.

- [ ] **Step 3: The app opens and reports healthy**

Run: `uv run omnilingual-ui --no-window --port 8766 &`
Then: `curl -s http://127.0.0.1:8766/api/health`
Expected: `ok: true` with `arch`, `macos`, `arm64`, `python` populated.
Then open `http://127.0.0.1:8766/` in a browser and confirm the panel renders with every control from the spec's §9 table. This is criteria 1 and the manual part of §14.
Kill the process afterwards.

- [ ] **Step 4: A live session streams rows while the file grows**

With the `Omnilingual` Aggregate Device present: start a live run from the panel and speak for at least a minute.
Expected: rows appear in the table as chunks are sealed; `standup.md` grows alongside; on stop, the page's segment count and the count of transcript blocks in `standup.md` agree, and the `end` message reports `done`. This is criteria 2 and 3.

If the device is absent on this machine, record that as an environment limitation and rely on `tests/ui/test_session.py::test_live_run_emits_state_segment_status_and_end`, which asserts the same message ordering and reports the session directory.

- [ ] **Step 5: Stop leaves a valid file**

Confirm the `standup.md` from Step 4 has no `· growing` marker and opens cleanly in a Markdown viewer.
Expected: a valid file. This is part of criteria 3.

- [ ] **Step 6: A recording renders identically to the CLI**

Transcribe the same recording twice, once through the UI and once through the CLI, with identical parameters, writing to different paths.

Run:
```bash
uv run omnilingual transcribe sample.m4a --out /tmp/cli.md --stt groq --mt gemini
```
and the same request through `POST /api/session/start` with `mode: recording` and `out: /tmp/ui.md`.
Expected: `diff /tmp/cli.md /tmp/ui.md` is empty. This is criteria 4, and it is the real test of the claim that the UI renders nothing itself.

If no API keys are configured, `tests/ui/test_session.py::test_recording_run_writes_a_rendered_transcript` plus the existing `tests/render/test_markdown.py` cover the renderer; record that substitution.

- [ ] **Step 7: Halting on cost cap is recoverable**

Start a live run with `--max-cost` set to a value the run will exceed, using the panel's cost-cap field.
Expected: status becomes `stopping`/`done` with `recoverable: true`, the page offers a **Recover this session** button, and pressing it completes the transcript from the session directory without re-billing sealed chunks.
This is criteria 5.

`tests/pipeline/test_run_live.py::test_cost_cap_halts_api_but_keeps_sealing` covers the halt itself; the UI-side recovery button is covered by `test_page_offers_recovery_only_when_the_run_is_recoverable`.

- [ ] **Step 8: The audio banner appears and Set up audio works**

Run the app with the Aggregate Device removed (destroy it via the helper's `destroy` subcommand), then restart the app.
Expected: the banner names what is missing, and pressing **Set up audio** recreates the device and clears the banner.
This is criteria 6.

Do not press **Restart audio daemon** on a machine where you do not want the macOS authorisation prompt; it is covered by `tests/ui/test_audio.py::test_restart_daemon_uses_the_mac_authorisation_prompt`.

- [ ] **Step 9: Confirm no key value can be observed**

Run: `curl -s -H "X-Omnilingual-Token: $(curl -s http://127.0.0.1:8766/token.js | sed "s/.*= '//;s/';//")" -H "Host: 127.0.0.1:8766" http://127.0.0.1:8766/api/keys`
Expected: only booleans. Then confirm `.env` is still mode `0600` and `~/.config/omnilingual/ui.toml` is `0600`.

- [ ] **Step 10: Confirm the working tree is clean of strays**

Run: `git status --short`
Expected: only the intended changes. Remove any temporary directories created during verification.

- [ ] **Step 11: Commit any fixes this task surfaced

If Steps 1-10 required a fix, commit it in the task that owns the affected file, with a message describing the defect rather than the plan. If nothing needed fixing, record that no commit was necessary.

---

## Self-Review

**Spec coverage.** Every phase 1 section maps to a task: §4 (the `run_live` change) → Task 3; §5 (package layout) → Tasks 2 and 4-9; §6 (in-process thread) → Task 6; §7 (routes, WebSocket, hardening) → Task 7, hardened end to end in Task 8's page; §8 (run lifecycle, one run, 409, disconnect-safe) → Tasks 6-7; §9 (parameters and shared validation) → Tasks 1, 4, 7, 8; §10 (settings and secrets) → Task 4; §11 (audio readiness, never auto-route output) → Tasks 5, 7, 8; §13 (error handling) → Tasks 6-7 plus the log strip in Task 8; §14 (testing) → a test file in every task, none needing hardware; §15 (dependencies and packaging) → Task 2; §16 (phase 1 acceptance, and the `setup-mac.sh` follow-up) → Tasks 10 and 11.

Deliberately not in this plan, per spec §16: the phase 2 Setup screen (its own plan), the CLI reading `ui.toml`, multi-window sessions, and editing a transcript from the UI.

**Type consistency.** The plan introduces no parallel options type: a request is validated into the package's own `Settings` and `LiveOptions`, so the UI cannot drift from what the CLI accepts. `SessionRunner.start_live`, `start_recording`, and `recover` are called only from `server._build` and `post_recover`, and their keyword arguments match those call sites exactly. `SessionRunner.segment_message` is the single place a transcript row is shaped, and its output keys match the spec's §7.1 shape and the `addRow` reader in `app.js` field for field. `AppState.queue` is one queue, drained by whichever WebSocket is connected. `audio.Readiness`, `settings.DEFAULTS`, and `secrets.present()` are each defined once and consumed by both the server and its tests. Every function a test imports is declared in its task's Interfaces block.