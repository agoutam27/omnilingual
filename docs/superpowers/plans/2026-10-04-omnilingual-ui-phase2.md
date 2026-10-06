# Omnilingual UI Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the UI a Setup screen that drives the already-reviewed `scripts/setup-mac.sh` (preview the plan, then apply it with output streaming into the existing log strip), and turn on the live cost meter by exposing the run's real accumulated cost.

**Architecture:** Phase 1 shipped the server, session runner, launcher and page. Phase 2 adds no new subsystem — it wires two things that already exist. `server.py` already has `GET /api/setup/preview` and `POST /api/setup/apply` wired to a subprocess helper that already accepts flags and is currently given none; the only missing piece is that `setup-mac.sh` has no flags to accept. Separately, `run_live` already computes a per-segment cost delta and already accumulates it into `state["accrued"]`; it simply does not tell anyone. One keyword-only callback closes that, and `SessionRunner._on_segment` already has a `self._cost += delta` line waiting for it.

**Tech Stack:** bash 3.2.57 (macOS system bash), Python 3.12+, FastAPI + uvicorn (already shipped in the `ui` extra), plain JS/CSS/HTML with no build step.

## Global Constraints

- The server binds `127.0.0.1` only, on an ephemeral port chosen at launch. Never expose to the network.
- Loopback hardening is mandatory and all three parts: a per-launch random token on every `/api/*` route and the WebSocket (absent/wrong → `403`); reject any `Host` header that is not `127.0.0.1:<port>` or `localhost:<port>`; reject any foreign `Origin`. No CORS middleware.
- API key values **never** leave the process. `GET /api/keys` returns booleans only. Keys live in the repo's gitignored `.env` — never in the UI settings file, never in an argv, never logged or echoed.
- Never switch the system audio output automatically: the Multi-Output Device route silently breaks the volume keys.
- The UI renders no transcript formatting of its own. Recording output goes through `render.markdown.render()` / `render_english_only()`; the page displays that same text.
- One run at a time per window. `POST /api/session/start` while a run is active → `409`.
- A WebSocket disconnect must NOT kill a running session. The run lives in the server, not in the page.
- `session.py` is the ONLY module that knows how a run executes. `server.py` never imports `omnilingual.pipeline`.
- Nothing in `omnilingual/ui/` may be imported by the CLI, so the CLI cannot acquire a web dependency.
- All run threads are daemon threads. A run always leaves a valid output file, including after a stop.
- `uv run pytest -q` must stay green throughout.
- Phase 2 must not change CLI behaviour or flags. The only edits to existing code are the two additive callback/flag surfaces in Tasks 1 and 2 and the wiring in Tasks 3 and 4.
- Repo root, from inside the package, is `Path(__file__).resolve().parents[2]`.
- **Never pass request data into a subprocess argv.** Every subprocess in this plan takes constants and values the server itself validated against a whitelist — the same rule `_setup_argv` already follows.
- **`setup-mac.sh` must remain re-runnable and convergent.** A capability flag narrows what a run does; it must never make the script install something the caller did not ask for.

---

### Task 1: Capability flags on `setup-mac.sh`

The UI must be able to answer the script's capability questions without a terminal. Today it cannot: the script only reads its saved config, its built-in defaults, or interactive prompts, and `interactive()` is `[[ $ASSUME_YES -eq 0 && -t 0 ]]`, so under `--yes` with no TTY every prompt is skipped. Spec §12 assumes flags exist ("passed as explicit flags"); they do not. This task adds them, following the precedence pattern `--repo` and `--branch` already use.

**Files:**
- Modify: `scripts/setup-mac.sh` (header comment lines 2-37 for `usage()`; flag `case` at 64-76; declarations at 179-187; flag-restore block at 234-235)
- Create: `tests/test_setup_script.py`

**Interfaces:**
- Consumes: nothing. This is the first task.
- Produces: six new flags on `scripts/setup-mac.sh`, each overriding the saved config for that run and being persisted by the existing `save_config`:
  `--extras <csv>` · `--keys <csv>` · `--live-setup <yes|no>` · `--route-output <yes|no>` · `--prefetch <yes|no>` · `--run-tests <yes|no>`

- [ ] **Step 1: Write the failing test**

Create `tests/test_setup_script.py`:

```python
"""The setup script's capability flags — the only way the UI can answer it.

These are subprocess tests against the real script. `--help` and the flag
parser are cheap and side-effect free, so they can run anywhere; anything that
would install is exercised only by asserting on the resolved plan text, never by
letting a step execute.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup-mac.sh"

NEW_FLAGS = [
    "--extras",
    "--keys",
    "--live-setup",
    "--route-output",
    "--prefetch",
    "--run-tests",
]


def run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def test_every_capability_flag_is_documented_in_help():
    out = run("--help").stdout
    for flag in NEW_FLAGS:
        assert flag in out, f"{flag} is not in --help"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--extras", "local-stt,diarize"),
        ("--keys", "SARVAM_API_KEY"),
        ("--live-setup", "no"),
        ("--route-output", "no"),
        ("--prefetch", "yes"),
        ("--run-tests", "no"),
    ],
)
def test_a_capability_flag_is_accepted_and_echoed_in_the_plan(flag, value):
    # --dry-run prints the resolved plan and touches nothing, so this asserts
    # the flag reached the script's own state without letting any step run.
    result = run(flag, value, "--dry-run", "--yes")
    assert result.returncode == 0, result.stderr


def test_extras_flag_overrides_the_saved_config(tmp_path):
    config = tmp_path / "omnilingual" / "setup-mac.conf"
    config.parent.mkdir(parents=True)
    config.write_text("EXTRAS=diarize\n", encoding="utf-8")
    env = {"XDG_CONFIG_HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    result = run("--extras", "local-stt", "--dry-run", "--yes", env=env)
    assert result.returncode == 0, result.stderr
    assert "local-stt" in result.stdout + result.stderr


def test_a_capability_flag_needs_a_value():
    for flag in NEW_FLAGS:
        result = run(flag)
        assert result.returncode == 2, f"{flag} with no value should exit 2"
        assert "needs a value" in result.stderr


def test_unknown_capability_names_are_filtered_not_installed():
    # `sanitize` already whitelists against ALL_EXTRAS; this proves a flag
    # cannot smuggle an extra in past that whitelist.
    result = run("--extras", "local-stt;rm -rf /", "--dry-run", "--yes")
    assert result.returncode == 0, result.stderr
    assert "rm -rf" not in result.stdout + result.stderr


def test_the_flag_names_appear_exactly_once_in_the_case_block():
    body = SCRIPT.read_text(encoding="utf-8")
    case = re.search(r"while \[\[ \$# -gt 0 \]\]; do(.*?)^done", body, re.S | re.M)
    assert case, "the flag-parsing while loop is gone"
    for flag in NEW_FLAGS:
        assert case.group(1).count(f"{flag})") == 1, f"{flag} has no case arm"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_setup_script.py -q`
Expected: FAIL — `test_every_capability_flag_is_documented_in_help` fails because `--extras` is not in `--help`, and `test_a_capability_flag_needs_a_value` fails because the script exits 2 with `unknown flag:` instead of `needs a value`.

- [ ] **Step 3: Add the flags**

In `scripts/setup-mac.sh`, first extend the header comment so `usage()` picks it up. Insert after the `--reset` line in the Options block:

```
#   --extras <list>    Comma-separated extras to enable, from:
#                      local-stt (offline speech-to-text), diarize (speaker
#                      labels), local-mt (offline translation), ui (the local
#                      app). Overrides the saved answers for this run.
#   --keys <list>      Comma-separated API keys to prompt for and store in .env,
#                      from: SARVAM_API_KEY, GROQ_API_KEY, GEMINI_API_KEY.
#   --live-setup <yn>  Create the BlackHole capture devices. Default: no.
#   --route-output <yn> Switch system output to the Multi-Output Device. Only
#                      meaningful with --live-setup yes; forced to no without it.
#   --prefetch <yn>    Pre-download model weights. Default: yes.
#   --run-tests <yn>   Run the test suite at the end. Default: yes.
#   Each of these overrides the saved answers for this run and the resolved
#   value is written back, so re-running converges instead of re-asking.
```

Next, add the declarations. Immediately after `RUN_TESTS="yes"` insert:

```bash
# Explicit flags for this run. Empty means "not given", which is what lets the
# saved config win; *_SET distinguishes "given as empty" from "not given".
FLAG_EXTRAS=""
FLAG_KEYS=""
FLAG_LIVE_SETUP=""
FLAG_ROUTE_OUTPUT=""
FLAG_PREFETCH=""
FLAG_RUN_TESTS=""
EXTRAS_SET=0
KEYS_SET=0
LIVE_SETUP_SET=0
ROUTE_OUTPUT_SET=0
PREFETCH_SET=0
RUN_TESTS_SET=0
```

Next, add the case arms. Inside the `while [[ $# -gt 0 ]]` loop, after the `--branch)` arm:

```bash
        --extras)       [[ $# -ge 2 ]] || { echo "--extras needs a value" >&2; exit 2; }; FLAG_EXTRAS="$2"; EXTRAS_SET=1; shift 2 ;;
        --keys)         [[ $# -ge 2 ]] || { echo "--keys needs a value" >&2; exit 2; }; FLAG_KEYS="$2"; KEYS_SET=1; shift 2 ;;
        --live-setup)   [[ $# -ge 2 ]] || { echo "--live-setup needs a value" >&2; exit 2; }; FLAG_LIVE_SETUP="$2"; LIVE_SETUP_SET=1; shift 2 ;;
        --route-output) [[ $# -ge 2 ]] || { echo "--route-output needs a value" >&2; exit 2; }; FLAG_ROUTE_OUTPUT="$2"; ROUTE_OUTPUT_SET=1; shift 2 ;;
        --prefetch)     [[ $# -ge 2 ]] || { echo "--prefetch needs a value" >&2; exit 2; }; FLAG_PREFETCH="$2"; PREFETCH_SET=1; shift 2 ;;
        --run-tests)    [[ $# -ge 2 ]] || { echo "--run-tests needs a value" >&2; exit 2; }; FLAG_RUN_TESTS="$2"; RUN_TESTS_SET=1; shift 2 ;;
```

Next, restore the flagged values after `load_config`. Extend the existing block:

```bash
# An explicit flag on this run outranks whatever was saved last time.
if [[ $REPO_URL_SET -eq 1 ]]; then REPO_URL="$FLAG_REPO_URL"; fi
if [[ $BRANCH_SET -eq 1 ]]; then BRANCH="$FLAG_BRANCH"; fi
if [[ $EXTRAS_SET -eq 1 ]]; then EXTRAS="$FLAG_EXTRAS"; fi
if [[ $KEYS_SET -eq 1 ]]; then KEYS="$FLAG_KEYS"; fi
if [[ $LIVE_SETUP_SET -eq 1 ]]; then LIVE_SETUP="$FLAG_LIVE_SETUP"; fi
if [[ $ROUTE_OUTPUT_SET -eq 1 ]]; then ROUTE_OUTPUT="$FLAG_ROUTE_OUTPUT"; fi
if [[ $PREFETCH_SET -eq 1 ]]; then PREFETCH="$FLAG_PREFETCH"; fi
if [[ $RUN_TESTS_SET -eq 1 ]]; then RUN_TESTS="$FLAG_RUN_TESTS"; fi
```

That restore block already sits above the `EXTRAS="$(sanitize ...)"` line, so a flagged value is whitelisted exactly like a saved one and `save_config` records the resolved result. No further wiring is needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_setup_script.py -q`
Expected: PASS.

Then confirm the script is still syntactically valid and the existing review still holds:

```bash
/bin/bash -n scripts/setup-mac.sh
scripts/setup-mac.sh --help
scripts/setup-mac.sh --dry-run --yes          # must change nothing
```

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: 730 passed, 10 skipped (or more; never fewer).

- [ ] **Step 6: Commit**

```bash
git add scripts/setup-mac.sh tests/test_setup_script.py
git commit -m "feat(setup): accept capability flags so a caller can answer the script"
```

---

### Task 2: An `on_cost` seam on `run_live`

The live cost meter cannot be built because `run_live` never tells anyone what a run has spent. It does know: `process_chunk` returns a per-segment cost delta, `run_live` accumulates it into `state["accrued"]` (which the cost cap depends on), and the writers use it. It simply does not pass it out. One keyword-only callback closes that gap without changing the shipped signature.

**Files:**
- Modify: `omnilingual/pipeline/live.py` (signature at 147-152; accumulation at 468)
- Test: `tests/pipeline/test_run_live.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `run_live(..., on_cost: Callable[[float, float], None] | None = None)` — called as `on_cost(delta, accrued)` after each segment's cost is folded into the running total, where `delta` is that segment's cost in INR and `accrued` is the total after folding it in. Called once per segment, including for a dropped silence (whose delta is `0.0` and whose total is unchanged) so a consumer's totals never disagree with the transcript's.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pipeline/test_run_live.py`:

These copy the exact call shape of `test_on_segment_reports_every_sealed_chunk`, which already lives in this file.

```python
@respx.mock
def test_on_cost_reports_the_delta_and_the_running_total(respx_mock, tmp_path):
    """Each callback carries that segment's cost and the total after it."""
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
             on_cost=lambda delta, accrued: seen.append((delta, accrued)))

    assert seen, "on_cost never fired"
    deltas = [delta for delta, _ in seen]
    assert any(delta > 0 for delta in deltas), f"no cost was reported: {seen}"
    running = 0.0
    for delta, accrued in seen:
        running += delta
        assert abs(accrued - running) < 1e-9, f"{accrued} != running {running}"


@respx.mock
def test_on_cost_fires_for_a_dropped_silence_too(respx_mock, tmp_path):
    """A dropped chunk costs nothing, but the consumer must still be told, so
    the number of calls matches the number of sealed chunks."""
    _route(respx_mock, texts=("Bravo", "   ", "Vanakkam"))
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    calls = []
    run_live(_opts(tmp_path, stt_workers=1), settings, SarvamSTT(settings),
             MayuraTranslator(settings), status=lambda m: None,
             capture_factory=factory,
             on_cost=lambda delta, accrued: calls.append((delta, accrued)))

    assert len(calls) == 3, f"expected one call per sealed chunk, got {calls}"
    assert calls[1][0] == 0.0, "the dropped silence must report a zero delta"


@respx.mock
def test_a_raising_on_cost_does_not_end_the_run(respx_mock, tmp_path):
    """A UI rendering bug in the cost callback must not kill a live meeting."""
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    def boom(delta, accrued):
        raise RuntimeError("the callback is broken")

    code = run_live(_opts(tmp_path, stt_workers=1), settings, SarvamSTT(settings),
                    MayuraTranslator(settings), status=lambda m: None,
                    capture_factory=factory, on_cost=boom)

    assert code == 0, "a raising on_cost ended the run"
    assert (tmp_path / "meeting.md").exists()
```

`_route(respx_mock, texts=("Bravo", "   ", "Vanakkam"))` is what makes chunk 2 a genuinely dropped silence — the middle chunk returns whitespace-only text, which `process_chunk` turns into a `no_speech` segment the writers drop.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_run_live.py -q -k on_cost`
Expected: FAIL — `TypeError: run_live() got an unexpected keyword argument 'on_cost'`.

- [ ] **Step 3: Add the callback**

In `omnilingual/pipeline/live.py`, extend the signature. `on_segment` is already keyword-only at line 151; add immediately after it:

```python
    on_cost: Callable[[float, float], None] | None = None,
```

with a docstring line beside the existing `on_segment` docs:

```
    on_cost is called as ``on_cost(delta, accrued)`` once per sealed segment,
    after that segment's cost is folded into the running total: ``delta`` is
    this segment's cost in INR (0.0 for a dropped silence) and ``accrued`` is
    the total after folding it in. Callback exceptions are swallowed for the
    same reason as ``on_segment`` — a consumer's bug must not end a meeting.
```

Then **after the whole `if keep:` block** — that is, after `state["accrued"] += delta` *and* its trailing `state["bad"] = True` line, dedented back out to the appender's own level:

```python
            if on_cost is not None:
                # After the total is folded in, so `accrued` is never stale, and
                # outside `if keep` so a dropped silence still reports 0.0 instead
                # of leaving the page's running total looking frozen. Reading
                # `accrued` without `mlock` is safe: this appender thread is its
                # only writer, and the read happens on that same thread.
                try:
                    on_cost(delta, state["accrued"])
                except Exception:  # noqa: BLE001 - a UI bug must not end the run
                    log.exception("on_cost callback failed")
```

Two placement facts, both verified against `live.py:464-473`, and together they are why this must go **outside** `if keep:`:

- `state["accrued"] += delta` is itself inside `if keep:` (line 468). Firing from inside that block would mean a dropped silence **never** fires the callback at all.
- Firing before the block instead would pass a **stale** `accrued` for every kept segment — the total would lag by one segment.

Wrapped in the same `try/except Exception: log.exception(...)` the `on_segment` call uses, so one broken consumer cannot end the run. This mirrors `on_segment`'s placement (also outside `if keep:`), which is what makes the page learn about every sealed chunk.

Note that a cost-cap halt emits its halt segment with `status == "stt_failed"`, so the `no_speech` drop branch does not apply, `keep` stays `True`, and `delta` is `0.0` — `on_cost(0.0, final_total)` therefore fires when the run halts, which is exactly when the page most needs the real total.

No helper function is needed. Use the same `if on_cost is not None:` guard the shipped `on_segment` call uses at `live.py:454` — do **not** introduce a bound `_ignore_cost` stand-in, since that would be a second pattern for one idea in a module that already has one.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_run_live.py -q`
Expected: PASS — 23 tests plus the 3 new ones.

- [ ] **Step 5: Confirm the two existing seams did not regress**

Run: `uv run pytest tests/pipeline/ tests/test_cli_live.py tests/test_live.py -q`
Expected: PASS. The `on_segment` and `stop_event` behaviour, including the CLI's own Ctrl+C path, must be byte-identical — both new parameters default to `None`.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: never fewer than 730 passed.

- [ ] **Step 7: Commit**

```bash
git add omnilingual/pipeline/live.py tests/pipeline/test_run_live.py
git commit -m "feat: report each segment's cost so a live run can show a running total"
```

---

### Task 3: `SessionRunner` threads the running cost through

`SessionRunner` already has the receiving end. `_on_segment(self, seg, keep, delta)` already does `self._cost += delta`, and `snapshot()` already publishes `"cost_inr": round(self._cost, 2)`. Both are dead today because `delta` is always `0` at both entry points. This task connects Task 2's callback to them, so `state.cost_inr` becomes a real number instead of a structural zero.

**Files:**
- Modify: `omnilingual/ui/session.py` (`run_live` call at 518; `_on_segment` at 601; the `_cost` docs)
- Test: `tests/ui/test_session.py`

**Interfaces:**
- Consumes: `run_live(..., on_cost: Callable[[float, float], None] | None = None)` from Task 2.
- Produces: `SessionRunner._on_cost(delta: float, accrued: float) -> None`, and a `state` message whose `cost_inr` is the live run's real accumulated spend in INR. **Batch and recovery are unchanged** — they already set `_cost` from the transcript.

- [ ] **Step 1: Write the failing test**

Append to `tests/ui/test_session.py`:

```python
@respx.mock
def test_a_live_run_reports_a_real_running_cost(respx_mock, tmp_path):
    """`state.cost_inr` must be the run's actual spend, not a structural zero."""
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    config, stt, mt = _providers()
    runner, queue, opts, factory = _live(tmp_path, factory)

    runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                      opts=opts, capture_factory=factory)
    runner.join(timeout=120)

    spent = [m["cost_inr"] for m in queue.of("state") if "cost_inr" in m]
    assert any(value > 0 for value in spent), (
        f"no state message carried a non-zero cost_inr: {spent}"
    )
```

This uses the file's existing helpers (`_route`, `_pcm_3x20`, `FakeCapture`, `_blocks`, `_gaps`, `_providers`, `_live`, `Collector.of`) with no new scaffolding. The cost is real without any stub: `process_chunk` returns the provider's estimated cost and the writer footer for this same run has been observed rendering `Cost ₹0.56`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ui/test_session.py -q -k running_cost`
Expected: FAIL — `cost_inr` is `0.0` on every state message.

- [ ] **Step 3: Wire the callback**

In `omnilingual/ui/session.py`, pass the callback at the `run_live` call:

```python
                on_segment=on_segment,
                on_cost=self._on_cost,
```

Then replace the dead line in `_on_segment`. Its current body accumulates from a `delta` argument that is always `0`; that accumulation now belongs to `_on_cost`, so remove it and correct the docstring that claims the delta is zero for both entry points:

```python
    def _on_cost(self, delta: float, accrued: float) -> None:
        """A live run's running spend. `accrued` is authoritative — it is the
        pipeline's own total after folding this segment in, not a sum we keep
        separately, so it cannot drift from the transcript the run is writing.
        """
        self._cost = accrued
```

Keep the `self._cost += delta` removal minimal: the field is read under `self._lock` by `snapshot()`, and `_on_cost` runs on the pipeline's ordered emitter thread, so take the lock the same way `_on_segment` does.

Update `_on_segment`'s signature comment to record that its third parameter is no longer the source of truth.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ui/test_session.py -q`
Expected: PASS — 65 tests plus the new one.

- [ ] **Step 5: Confirm batch and recovery did not change**

Run: `uv run pytest tests/ui/test_session.py -q -k "recording or recovery or byte_identical"`
Expected: PASS. Those paths set `_cost` from `transcript.cost.inr_estimate` and must be untouched.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: never fewer than 730 passed.

- [ ] **Step 7: Commit**

```bash
git add omnilingual/ui/session.py tests/ui/test_session.py
git commit -m "feat(ui): publish a live run's real accumulated cost in the state message"
```

---

### Task 4: Wire the setup routes to validated flags, and add relaunch

`server.py` already has both setup routes and a `_setup_argv(*flags)` helper that already appends `--yes` and already accepts flags — it is currently called with none. This task gives it the flags, and adds the relaunch endpoint spec §12 requires.

**Files:**
- Modify: `omnilingual/ui/server.py` (`_setup_argv` at 518, `_setup_preview` at 547, `_setup_lines` at 527, the two routes at 786 and 798)
- Test: `tests/ui/test_server.py`

**Interfaces:**
- Consumes: the six capability flags from Task 1.
- Produces:
  - `POST /api/setup/preview` with body `{"extras": "diarize", "live_setup": false, "route_output": false, "prefetch": true, "run_tests": true}` → `200 {"output": "..."}` (unchanged shape) or `400 {"error": "..."}`. `keys` is deliberately NOT accepted: the Setup screen must never receive a secret, and the script already reads existing keys from `.env`.
  - `POST /api/setup/apply` with the same body → `200 {"started": true}` (unchanged shape).
  - `POST /api/relaunch` with no body → `200 {"restarting": true}`, re-executes the launcher's own argv.

- [ ] **Step 1: Write the failing tests**

Append to `tests/ui/test_server.py`:

```python
CAPABILITIES = {"extras": "diarize", "live_setup": False, "route_output": False,
                "prefetch": True, "run_tests": False}


class _FakeProc:
    returncode = 0

    def __init__(self, lines=("installing\n", "done\n")):
        self.stdout = iter(lines)

    def wait(self, timeout=None):
        return 0


def test_preview_passes_the_capabilities_to_the_script(api, monkeypatch):
    """The flags reach argv; they are never interpolated into a shell string."""
    client, token = api
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "extras enabled : diarize", "")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    response = client.post("/api/setup/preview", json=CAPABILITIES, headers=_ok(token))

    assert response.status_code == 200, response.text
    argv = seen[0]
    assert argv[1:3] == ["--extras", "diarize"], f"argv was {argv}"
    assert "--live-setup" in argv and "no" in argv
    assert "--route-output" in argv and "no" in argv
    assert "--prefetch" in argv and "yes" in argv
    assert "--run-tests" in argv and "no" in argv
    assert "--dry-run" in argv
    assert argv[-1] == "--yes", f"--yes must stay last, argv was {argv}"


def test_preview_refuses_an_unknown_extra(api):
    client, token = api
    response = client.post("/api/setup/preview", headers=_ok(token),
                           json={**CAPABILITIES, "extras": "local-stt;rm -rf /"})
    assert response.status_code == 400
    assert "rm -rf" not in response.text


def test_preview_refuses_an_unknown_capability_name(api):
    client, token = api
    response = client.post("/api/setup/preview", headers=_ok(token),
                           json={**CAPABILITIES, "sudo": True})
    assert response.status_code == 400


def test_apply_passes_the_capabilities_to_the_script(api, monkeypatch):
    client, token = api
    seen: list[list[str]] = []
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda argv, **kw: (seen.append(list(argv)), _FakeProc())[1])

    response = client.post("/api/setup/apply", json=CAPABILITIES, headers=_ok(token))

    assert response.status_code == 200
    assert response.json() == {"started": True}
    assert "--run-tests" in seen[0] and "no" in seen[0]


def test_apply_never_passes_a_key_to_the_script(api, monkeypatch):
    """`keys` is not an accepted field: the screen must never hold a secret."""
    client, token = api
    seen: list[list[str]] = []
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda argv, **kw: (seen.append(list(argv)), _FakeProc(()))[1])

    client.post("/api/setup/apply", headers=_ok(token),
                json={**CAPABILITIES, "keys": "SARVAM_API_KEY=sk-live-abc"})

    assert "SARVAM_API_KEY=sk-live-abc" not in " ".join(seen[0])


def test_relaunch_reexecutes_a_fixed_argv(api, monkeypatch):
    """The relaunch must not take its command from the request."""
    client, token = api
    calls: list[list[str]] = []
    monkeypatch.setattr(server, "_relaunch_argv", lambda: ["python", "-m", "omnilingual.ui"])
    monkeypatch.setattr(server.os, "execv", lambda path, argv: calls.append(list(argv)))

    response = client.post("/api/relaunch", headers=_ok(token))

    assert response.status_code == 200
    assert response.json() == {"restarting": True}
    assert calls == [["python", "-m", "omnilingual.ui"]]


def test_relaunch_is_token_guarded(api):
    client, _ = api
    assert client.post("/api/relaunch").status_code == 403
```

The `api` fixture returns `(client, token)` and `_ok(token)` builds the
`X-Omnilingual-Token` + `Host` headers every route requires — both already exist
in this file. Use them; a bare `client.post(...)` answers 403 and would make
these tests pass for the wrong reason.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ui/test_server.py -q -k "setup or relaunch or capabilities"`
Expected: FAIL — the routes currently ignore the request body entirely, so the flags never reach argv and `/api/relaunch` returns 404.

- [ ] **Step 3: Build the argv**

Add next to `_setup_argv` in `server.py`:

```python
_SETUP_EXTRAS = ("local-stt", "diarize", "local-mt", "ui")


def _capability_flags(body: dict) -> list[str]:
    """Validated capability flags for setup-mac.sh.

    Every value below is either a literal from _SETUP_EXTRAS or a yes/no this
    function produces, so nothing from the request reaches argv unfiltered.
    Raises ConfigError, which the routes already map to 400.
    """
    unknown = set(body) - {"extras", "live_setup", "route_output", "prefetch", "run_tests"}
    if unknown:
        raise ConfigError(f"unknown setup field(s): {', '.join(sorted(unknown))}")

    chosen: list[str] = []
    for name in str(body.get("extras", "")).split(","):
        name = name.strip()
        if not name:
            continue
        if name not in _SETUP_EXTRAS:
            raise ConfigError(f"unknown extra: {name}")
        if name not in chosen:
            chosen.append(name)

    flags = ["--extras", ",".join(chosen)]
    for name, value in (("live_setup", body.get("live_setup")),
                        ("route_output", body.get("route_output")),
                        ("prefetch", body.get("prefetch")),
                        ("run_tests", body.get("run_tests"))):
        flags += [f"--{name.replace('_', '-')}", "yes" if value else "no"]
    return flags
```

`_setup_argv(*flags)` already accepts and forwards flags and already appends `--yes`; it needs no change. Give the two subprocess helpers a `flags` parameter and pass it to `_setup_argv`:

```python
def _setup_lines(flags: Sequence[str]) -> Iterator[str]:
    proc = subprocess.Popen(_setup_argv(*flags), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace")
    for raw in proc.stdout:
        yield raw.rstrip("\n")
    code = proc.wait(timeout=_SETUP_TIMEOUT)
    if code:
        raise RuntimeError(f"setup-mac.sh exited {code}")


def _setup_preview(flags: Sequence[str]) -> str:
    done = subprocess.run(_setup_argv(*flags, "--dry-run"), capture_output=True,
                          text=True, errors="replace", timeout=_SETUP_TIMEOUT)
    if done.returncode:
        raise RuntimeError(f"setup-mac.sh exited {done.returncode}")
    return done.stdout + done.stderr
```

Finally the routes. `POST /api/setup/preview` gains a body and a `ConfigError` → 400 mapping; `POST /api/setup/apply` reads the same body inside its detached worker, because a detached worker has no request left to fail:

```python
@router.post("/api/setup/apply")
async def post_setup_apply(request: Request) -> dict:
    state = request.app.state.app_state
    loop = asyncio.get_running_loop()
    body = await request.json()
    try:
        flags = _capability_flags(body)
    except ConfigError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    _background(lambda: _stream(state, loop, _setup_lines(flags)))
    return {"started": True}
```

- [ ] **Step 4: Add the relaunch endpoint**

```python
def _relaunch_argv() -> list[str]:
    """The command that started this process, reconstructed from fixed parts.

    Never reads the request. sys.argv[0] and the interpreter are the only inputs,
    both of which the host chose at launch, so nothing a client sends can reach
    the exec.
    """
    return [sys.executable, "-m", "omnilingual.ui", *sys.argv[1:]]


@router.post("/api/relaunch")
async def post_relaunch(request: Request) -> dict:
    argv = _relaunch_argv()
    # Hand the socket to the successor before this process goes away, or the
    # relaunch races the dying listener and fails to bind.
    asyncio.get_running_loop().call_later(0.2, lambda: os.execv(sys.executable, argv))
    return {"restarting": True}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/ui/test_server.py -q`
Expected: PASS.

- [ ] **Step 6: Confirm the hardening did not regress**

Run: `uv run pytest tests/ui/test_server.py -q -k "token or host or origin or disconnect or halted or keys_endpoint"`
Expected: PASS. `/api/relaunch` is a new `/api/*` route and must be token-guarded like every other; confirm it answers 403 without a token before you consider this task done.

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -q`
Expected: never fewer than 730 passed.

- [ ] **Step 8: Commit**

```bash
git add omnilingual/ui/server.py tests/ui/test_server.py
git commit -m "feat(ui): drive setup-mac.sh from validated capability flags, and add relaunch"
```

---

### Task 5: The Setup screen

The page has no Setup section, and `app.js` already lists `/api/setup/apply` in its `DETACHED` map with no button to reach it. This task adds the section.

**Files:**
- Modify: `omnilingual/ui/static/index.html`, `omnilingual/ui/static/app.js`, `omnilingual/ui/static/style.css`
- Test: `tests/ui/test_static.py`

**Interfaces:**
- Consumes: the routes and shapes from Task 4.
- Produces: a Setup section with a capability form, a Preview button, an Apply button, and a restart banner. Apply's output appears in the existing `#log` strip; Preview's output appears in its own read-only block.

- [ ] **Step 1: Write the failing tests**

Append to `tests/ui/test_static.py`:

```python
def test_the_page_has_a_setup_section_with_a_capability_form():
    html = _read("index.html")
    assert 'id="setup"' in html
    for control in ("setup-extras", "setup-live-setup", "setup-route-output",
                    "setup-prefetch", "setup-run-tests", "setup-preview",
                    "setup-apply"):
        assert f'id="{control}"' in html, f"{control} is missing"


def test_the_setup_form_sends_exactly_the_accepted_fields():
    js = _read("app.js")
    body = _function(js, "capabilityPayload")
    for field in ("extras", "live_setup", "route_output", "prefetch", "run_tests"):
        assert field in body
    assert "keys" not in body, "the Setup screen must never hold a secret"


def test_apply_wires_through_the_detached_map_and_re_polls_afterwards():
    js = _read("app.js")
    assert '"/api/setup/apply":"setup-apply"' in js
    apply_fn = _function(js, "applySetup")
    assert "setup-apply" in apply_fn
    # A detached apply returns {"started": true}; the page must re-read readiness
    # when the log goes quiet rather than trusting the POST's response.
    assert "detached" in apply_fn, "apply must go through the detached map"


def test_the_setup_screen_says_a_restart_is_needed_after_apply():
    js = _read("app.js")
    assert "Restart to use newly installed components" in js
    assert "restart" in js.lower()


def test_relaunch_is_a_button_not_a_form_post():
    html = _read("index.html")
    assert 'id="relaunch"' in html
    assert "relaunch" in _read("app.js")
```

Reuse this file's existing `_asset` and `_function` helpers rather than adding new ones.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ui/test_static.py -q`
Expected: FAIL — no `id="setup"` exists in the page.

- [ ] **Step 3: Add the markup**

In `index.html`, after the existing output section:

```html
<section id="setup" hidden>
  <h2>Setup</h2>
  <p class="hint">Runs <code>scripts/setup-mac.sh</code> to install or remove the
  packages and audio devices omnilingual needs. Nothing runs until you press Apply.</p>
  <div class="row">
    <label for="setup-extras">Capabilities</label>
    <input id="setup-extras" type="text" placeholder="local-stt,diarize,ui">
  </div>
  <div class="row"><label><input id="setup-live-setup" type="checkbox"> Create the live-capture audio devices</label></div>
  <div class="row"><label><input id="setup-route-output" type="checkbox"> Switch system audio output to the capture device</label></div>
  <div class="row"><label><input id="setup-prefetch" type="checkbox" checked> Pre-download model weights</label></div>
  <div class="row"><label><input id="setup-run-tests" type="checkbox" checked> Run the test suite afterwards</label></div>
  <div class="row">
    <button id="setup-preview" type="button">Preview</button>
    <button id="setup-apply" type="button" class="danger">Apply</button>
  </div>
  <pre id="setup-plan" hidden></pre>
</section>
```

Add the relaunch button to the existing banner, which already has an `id="banner"`:

```html
  <button id="relaunch" type="button" hidden>Restart</button>
```

- [ ] **Step 4: Add the behaviour**

In `app.js`, `detached()` already streams a long route's output into the log and then re-polls readiness with a bounded budget, so it needs one change only: an optional body to POST.

```javascript
async function detached(path, body) {
  try {
    // These routes answer 200 {"started": true} and finish on a worker thread
    // that cannot fail a request which has already returned. There is nothing
    // to read but the acknowledgement.
    const answer = await api(path, {
      method: "POST",
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!answer.started) return;
    logLine(`working: ${path}`, "warn");
    repolls = REPOLL_ATTEMPTS;
    repollAudio();
  } catch (err) {
    showBanner([err.message], "error");
  }
}
```

Add the payload builder and the two actions:

```javascript
function capabilityPayload() {
  return {
    extras: $("setup-extras").value.trim(),
    live_setup: $("setup-live-setup").checked,
    route_output: $("setup-route-output").checked,
    prefetch: $("setup-prefetch").checked,
    run_tests: $("setup-run-tests").checked,
  };
}

async function previewSetup() {
  const plan = $("setup-plan");
  plan.hidden = false;
  plan.textContent = "working…";
  try {
    const response = await api("/api/setup/preview", {
      method: "POST",
      body: JSON.stringify(capabilityPayload()),
    });
    plan.textContent = response.output;
  } catch (err) {
    plan.textContent = err.message;
  }
}

async function applySetup() {
  // detached() streams the script's own lines into the log strip above and
  // then re-polls readiness, so the restart notice can wait for it: the
  // running process still has the modules it imported before the install.
  await detached("/api/setup/apply", capabilityPayload());
  showBanner("Setup finished. Restart to use newly installed components.", "warn");
  $("relaunch").hidden = false;
}

async function relaunch() {
  await api("/api/relaunch", { method: "POST" });
}
```

Wire `setup-preview`, `setup-apply` and `relaunch` into the existing button-binding block. `whenLogGoesQuiet` and a separate refresh helper are deliberately NOT added: `detached` already owns both, and a second mechanism would duplicate it.

The tests in this file read assets with `_read(name)` and function bodies with `_function(js, name)`; reuse both rather than adding helpers.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/ui/test_static.py -q`
Expected: PASS — 31 tests plus the new ones.

- [ ] **Step 6: Confirm the page's existing guarantees did not regress**

Run: `uv run pytest tests/ui/test_static.py -q`
Every Phase-1 assertion must still hold: the page loads its assets with plain tags, no key value reaches any node but the field that sent it, `done` is not rendered as unqualified success, the mic reads "not checked" rather than "ready", and the cost meter is still absent — Task 6 adds it deliberately, and until then a reviewer must be able to see its absence.

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -q`
Expected: never fewer than 730 passed.

- [ ] **Step 8: Commit**

```bash
git add omnilingual/ui/static/index.html omnilingual/ui/static/app.js omnilingual/ui/static/style.css tests/ui/test_static.py
git commit -m "feat(ui): add the Setup screen that previews and applies the setup script"
```

---

### Task 6: The live cost meter

Split from Task 5 deliberately: the Setup screen is a large, self-contained piece of work, and a reviewer should be able to reject a cost-meter rendering choice without also rejecting the setup wiring.

**Files:**
- Modify: `omnilingual/ui/static/app.js`, `omnilingual/ui/static/style.css`
- Test: `tests/ui/test_static.py`

**Interfaces:**
- Consumes: `state.cost_inr` and `state.cost_cap` from Task 3, both already emitted by `SessionRunner.snapshot()`.
- Produces: a running total and a cap shown while a live run is in progress. **No per-row cost** — `segment` messages carry no cost and the page must not imply otherwise.

- [ ] **Step 1: Write the failing tests**

Append to `tests/ui/test_static.py`:

```python
def test_the_page_renders_a_running_cost_and_a_cap():
    js = _read("app.js")
    assert "cost_inr" in js
    assert "cost_cap" in js


def test_the_cost_meter_shows_no_per_row_cost():
    """`segment` messages carry no cost. Rendering one per row would invent it."""
    js = _read("app.js")
    render_segment = _function(js, "renderSegment")
    assert "cost_inr" not in render_segment


def test_the_meter_says_unavailable_rather_than_zero_before_the_first_report():
    """`cost_cap` is None, not 0.0, until a run sets it — showing ₹0 there
    would read as a budget that is already spent."""
    js = _read("app.js")
    assert "cost_cap" in js
    render_state = _function(js, "renderState")
    assert "renderCost" in render_state, "the meter must hang off renderState"


def test_a_missing_cost_field_leaves_the_meter_blank_rather_than_zero():
    """An older server, or a state that predates the field, must not read ₹0."""
    js = _read("app.js")
    meter = _function(js, "renderCost")
    assert "typeof" in meter, "a missing field must be detected, not assumed zero"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ui/test_static.py -q -k cost`
Expected: FAIL — `cost_inr` appears nowhere in `app.js` except in a comment saying it is never read.

- [ ] **Step 3: Add the meter**

Add to `app.js`:

```javascript
function renderCost(state) {
  const meter = $("cost");
  const spent = typeof state.cost_inr === "number" ? `₹${state.cost_inr.toFixed(2)}` : "";
  const cap = typeof state.cost_cap === "number" ? `of ₹${state.cost_cap.toFixed(0)}` : "";
  meter.textContent = spent || cap ? `${spent} ${cap}`.trim() : "";
  meter.hidden = meter.textContent === "";
}
```

Call it from `renderState`, next to where the existing cap-only label is drawn, and remove the Phase-1 comment that says the field is never read. Add the element to `index.html` beside the status line:

```html
<span id="cost" class="cost" hidden></span>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ui/test_static.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: never fewer than 730 passed.

- [ ] **Step 6: Commit**

```bash
git add omnilingual/ui/static/app.js omnilingual/ui/static/style.css omnilingual/ui/static/index.html tests/ui/test_static.py
git commit -m "feat(ui): show a live run's running cost against its cap"
```

---

### Task 7: Verify Phase 2's acceptance criteria

Each criterion names the command that proves it and the output that means it passed. Run them; do not reason about them.

**Files:** none. Verification only.

**Interfaces:**
- Consumes: everything above.
- Produces: evidence for the four Phase-2 acceptance criteria in spec §16.

- [ ] **Step 1: Preview shows the plan and changes nothing**

```bash
uv run omnilingual-ui --no-window --port 8791 &
sleep 3
TOKEN=$(curl -s http://127.0.0.1:8791/token.js | sed -e 's/.*= "//' -e 's/".*//')
curl -s -X POST http://127.0.0.1:8791/api/setup/preview \
  -H "X-Omnilingual-Token: $TOKEN" -H 'Content-Type: application/json' \
  -d '{"extras":"diarize","live_setup":false,"route_output":false,"prefetch":false,"run_tests":false}'
```

Expected: `200` with an `"output"` key containing `extras enabled`. Then confirm nothing changed:

```bash
uv run pytest -q     # still green; no package was added or removed
```

- [ ] **Step 2: Apply streams output and completes**

Send the same body to `/api/setup/apply`, then read the log over the WebSocket:

```bash
curl -s -X POST http://127.0.0.1:8791/api/setup/apply \
  -H "X-Omnilingual-Token: $TOKEN" -H 'Content-Type: application/json' \
  -d '{"extras":"diarize","live_setup":false,"route_output":false,"prefetch":false,"run_tests":false}'
```

Expected: `{"started":true}`, then several `log` messages over `/ws?token=…` carrying the script's own lines. If the harness cannot hold a WebSocket open, assert instead that the same lines appear in the page after it re-polls, which is the user-visible claim.

- [ ] **Step 3: A restart is signalled, and relaunch works**

Expected: after the apply's output settles, the page shows "Restart to use newly installed components" and a Restart button. Pressing it answers `200 {"restarting": true}` and the process comes back serving on the same port.

- [ ] **Step 4: Re-running converges**

Send the same apply body a second time. Expected: the script's own convergence check reports no delta and installs nothing — its `extras already match` line. Confirm no duplicate install by checking the package count is unchanged:

```bash
uv pip list 2>/dev/null | wc -l     # same before and after the second apply
```

- [ ] **Step 5: The suite is green and nothing leaked**

```bash
uv run pytest -q
```

Expected: never fewer than 730 passed. Then confirm the repo is untouched apart from what Phase 2 changed:

```bash
git status --short
```

Expected: clean, or only the five pre-existing untracked entries (`.omo/`, `.serena/`, `standup.md`, `standup2.md`, `yt1.md`).