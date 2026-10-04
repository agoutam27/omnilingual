"""The ASGI app: routes, the WebSocket event stream, and the loopback hardening.

Deliberately knows nothing about transcription. It validates a request, hands the
panel payload to session.build_run — the one place a run is constructed — and
relays queued events to the page. Every request is validated before a run starts,
so a bad panel is a 400 rather than a session that dies halfway.

Four properties this module is responsible for, and why they are here rather than
in the layer above:

  * **A live start is refused when capture cannot work** (§9), naming the remedy.
    Answering 200 and failing seconds later leaves the user with an ffmpeg error
    and a panel that has stopped explaining itself.
  * **A run belongs to this process, not to the page.** The WebSocket is a reader.
    Closing the window mid-meeting must not destroy a transcription in progress, so
    nothing in the disconnect path touches a runner, and "is a run active" is
    answered by the runners' own status rather than by a flag the socket clears.
  * **No key value reaches a response.** PUT /api/keys reads the raw body rather
    than a declared model, because FastAPI echoes the submitted input inside a 422
    and a rejected value would come straight back out. Debug rendering stays off
    for the same reason: a traceback prints frame locals.
  * **The server never switches the audio output device.** It reports the current
    one, because a wrong output is a top cause of a live run recording nothing, and
    routing output to the Multi-Output Device breaks the user's volume keys.

session.py is the single module allowed to know how a run executes, so nothing here
reaches into the pipeline package and a test here fails if it ever does. fastapi is
confined to this file so the CLI keeps no web dependency.
"""

from __future__ import annotations

import asyncio
import platform
import secrets as pysecrets
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from omnilingual.audio.normalize import FfmpegMissingError, ensure_ffmpeg
from omnilingual.config import (
    DEFAULT_MT_MODELS,
    DEFAULT_STT_MODELS,
    MT_PROVIDERS,
    STT_PROVIDERS,
    ConfigError,
)

from omnilingual.ui import audio, secrets, settings
from omnilingual.ui.session import SessionRunner, build_run, live_options

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
SETUP_SCRIPT = REPO_ROOT / "scripts" / "setup-mac.sh"
UI_VERSION = "1"
TOKEN_HEADER = "X-Omnilingual-Token"

# The run states that mean the window is taken. 'done' is not one of them even when
# exit_code was 2: the file is written and some segments failed, which is a finished
# run. 'halted' is, because capture went on after the API calls stopped.
_BUSY_STATUSES = frozenset({"running", "stopping"})

# What a live panel's 'out' field means when the user cleared it, and what a
# recovery writes when the page named no output. A recording instead falls back to
# the recording's own name — see _build.
_DEFAULT_OUT = Path("standup.md")
_RECOVERED_OUT = Path("recovered.md")

# Same-origin, read-only, and needed before the page has a token, so they answer
# without one. Everything else under /api/ is token-guarded.
_UNGUARDED = frozenset({"/api/health", "/token.js", "/"})

# What a live start needs before it is worth answering 200, each with the fix the
# page offers next to it (§9). Named rather than reused from probe's `detail`,
# because this text is what the user is told to do, and probe's wording is
# diagnostic.
_LIVE_GATE = (
    ("device", "the Aggregate Device '{device}' does not exist — press Set up audio"),
    ("blackhole", "BlackHole 2ch is not installed — brew install blackhole-2ch"),
    ("ffmpeg", "ffmpeg is not on PATH — brew install ffmpeg"),
    ("ffprobe", "ffprobe is not on PATH — brew install ffmpeg"),
)
_MIC_REMEDY = ("macOS denied microphone access — approve this app in System "
               "Settings → Privacy & Security → Microphone")

_SETUP_TIMEOUT = 600

# What "/" answers until the page lands: 200 rather than a 404, because the window
# has to open on something and a stack trace is not a page.
_PLACEHOLDER = (
    "<!doctype html><meta charset=utf-8>"
    "<title>omnilingual</title>"
    "<p>The UI page has not landed yet; the API is up and token-guarded.</p>"
)


def mint_token() -> str:
    """A fresh, unguessable token per launch."""
    return pysecrets.token_urlsafe(32)


@dataclass
class AppState:
    """What one window owns. Lives on app.state.ui, so tests reach it from there."""

    token: str
    port: int
    runs: dict[str, SessionRunner] = field(default_factory=dict)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    loop: asyncio.AbstractEventLoop | None = None

    def free_run_id(self) -> str:
        return f"run-{len(self.runs) + 1}-{pysecrets.token_hex(4)}"

    def busy(self) -> SessionRunner | None:
        """The run holding the window, if any.

        Derived from the runners' own status instead of a flag, because a flag has
        to be cleared by whoever notices the end — and the end is noticed by the
        WebSocket. With the window closed, or closed again before it reconnected,
        nothing would ever clear it and the second run of a session would be
        refused forever.
        """
        return next((r for r in self.runs.values() if r.status in _BUSY_STATUSES),
                    None)


def _idle_state() -> dict:
    """The state snapshot for a window with no run behind it."""
    return {"run_id": None, "mode": "live", "status": "idle", "elapsed_s": 0.0,
            "cost_inr": 0.0, "cost_cap": None, "segments": 0, "dropped": 0,
            "out": None, "session_dir": None, "error": None}


# --- building a run --------------------------------------------------------


def _mode(body: Mapping) -> str:
    mode = str(body.get("mode", "live"))
    if mode not in ("live", "recording"):
        raise ConfigError(f"mode must be 'live' or 'recording', got {mode!r}")
    return mode


def _output_path(body: Mapping, *, fallback: Path = _DEFAULT_OUT) -> Path:
    """The transcript's path, with its directory created.

    A relative name resolves against the repo, not the process CWD: a window
    launched from Finder starts in /, and 'standup.md' would land there. The
    directory is created here so an unwritable path is a bad request instead of a
    run that fails after paying for transcription.
    """
    raw = str(body.get("out") or "").strip()
    path = Path(raw).expanduser() if raw else fallback
    path = path if path.is_absolute() else REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _recording_source(body: Mapping) -> Path:
    raw = str(body.get("source") or "").strip()
    if not raw:
        raise ConfigError("source is required in recording mode")
    source = Path(raw).expanduser()
    if not source.is_file():
        raise ConfigError(f"source not found: {source}")
    return source


def _build(body: Mapping) -> dict:
    """Validate a panel payload and return the kwargs the runner's start takes.

    Nothing about a run is decided here. build_run owns the providers, the field
    allowlist, the numeric coercion and the credential gate; live_options owns the
    mapping from panel field to LiveOptions field. A second copy of either is how
    the panel and the command line start disagreeing about what is legal.
    """
    mode = _mode(body)
    # Checked before the providers exist: a request that cannot start costs nothing.
    source = _recording_source(body) if mode == "recording" else None
    config, stt, translator, diarizer = build_run(body)
    providers = {"settings": config, "stt": stt, "translator": translator,
                 "diarizer": diarizer}
    if mode == "live":
        out = _output_path(body)
        return {"mode": mode,
                "kwargs": {**providers,
                           "opts": live_options(body, out=out)}}
    # cli.py's rule: --out defaults to the recording's own name, so two meetings
    # do not overwrite one another, and work_root is the panel's --work-dir with
    # the CLI's ".omnilingual beside the output" fallback.
    out = _output_path(body, fallback=source.with_suffix(".md"))
    stored = str(body.get("work_dir") or "")
    return {"mode": mode,
            "kwargs": {**providers, "source": source, "out": out,
                       "work_root": Path(stored).expanduser() if stored
                       else out.parent / ".omnilingual",
                       "english_only": bool(body.get("english_only"))}}


def _live_gate() -> None:
    """Refuse a live start whose capture cannot work, or raise ConfigError.

    The check the spec's §9 asks for and §13 lists as "start is refused in live
    mode". Answering 200 instead would start a thread that fails seconds later
    with an opaque ffmpeg error, long after the panel stopped being able to
    explain itself.

    Called after _build, never before: a rejected panel is answered from the
    payload alone and costs no hardware probe, so the user who mistyped a model
    id is told that rather than being sent to install BlackHole.

    The microphone is judged only when probe could actually ask. probe skips the
    one-second capture — and so reports False — whenever the device or ffmpeg is
    missing, and reading that as a denial would send the user to System Settings
    for a permission that was never the problem.

    The output device is deliberately not one of these checks. It is a warning
    the page already shows, not a refusal, and routing it here would break the
    volume keys.
    """
    readiness = audio.probe(mic=True)
    problems = [text.format(device=audio.DEVICE_NAME)
                for field, text in _LIVE_GATE if not getattr(readiness, field)]
    if readiness.ffmpeg and readiness.device and not readiness.mic_authorized:
        problems.append(_MIC_REMEDY)
    if problems:
        raise ConfigError("live capture is not ready: " + "; ".join(problems))


def _start(state: AppState, kwargs: dict, mode: str):
    loop = state.loop if state.loop and not state.loop.is_closed() else None
    runner = SessionRunner(run_id=state.free_run_id(), queue=state.queue,
                           loop=loop)
    state.runs[runner.run_id] = runner
    getattr(runner, {"live": "start_live", "recording": "start_recording",
                     "recover": "recover"}[mode])(**kwargs)
    return runner


# --- response bodies -------------------------------------------------------


def _readiness(*, mic: bool) -> dict:
    """The audio readiness shape, plus the two fields a page cannot infer.

    probe(mic=False) reports mic_authorized True because nothing was asked, and
    no field distinguishes that from a real answer; 'mic_checked' is what keeps a
    skipped probe from being read as a working microphone. The output-device
    warning is appended rather than acted on — the wrong output device is a top
    cause of "live capture records nothing", and switching it silently breaks the
    user's volume keys.
    """
    readiness = audio.probe(mic=mic)
    body = readiness.as_dict()
    body["mic_checked"] = mic
    detail = list(readiness.detail)
    if readiness.ok:
        if readiness.output is None:
            detail.append("the current output device could not be read")
        elif audio.DEVICE_NAME not in readiness.output:
            detail.append(
                f"the system output is '{readiness.output}'; live capture needs "
                f"the Multi-Output Device '{audio.DEVICE_NAME}', or it records "
                "nothing")
    body["detail"] = detail
    return body


async def _json_object(request: Request) -> dict:
    """The body as a JSON object, or a 400 that repeats none of it.

    A raw string is never echoed: a submitted key comes back in FastAPI's 422 body
    when a route declares a model, and a traceback prints frame locals, so both the
    message and this function's failure mode stay fixed text.
    """
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - any decode failure is one answer
        raise ConfigError("the request body is not valid JSON") from None
    if not isinstance(body, dict):
        raise ConfigError("the request body must be a JSON object")
    return body


# --- the setup script ------------------------------------------------------


def _setup_argv(*flags: str) -> list[str]:
    """argv for setup-mac.sh. Nothing from a request reaches it.

    An argv list, never a shell string: the script escalates to root through
    osascript, and the only strings here are constants.
    """
    return [str(SETUP_SCRIPT), *flags, "--yes"]


def _setup_apply(state: AppState) -> str:
    """Run setup-mac.sh for real, relaying each line as it appears.

    Line by line rather than one communicate() at the end, because the install runs
    for minutes and a page that shows nothing until it finishes cannot be trusted
    to still be open when it does. stderr is merged into stdout because the script
    writes everything human-facing there and keeps stdout for captured values.
    """
    lines: list[str] = []
    process = subprocess.Popen(_setup_argv(), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               errors="replace")
    try:
        for raw in process.stdout:
            line = raw.rstrip("\n")
            lines.append(line)
            state.queue.put_nowait({"type": "log", "level": "info",
                                    "message": line})
        code = process.wait(timeout=_SETUP_TIMEOUT)
    finally:
        process.stdout.close()
    if code != 0:
        raise RuntimeError(f"setup-mac.sh exited {code}")
    return "\n".join(lines)


def _setup_preview() -> str:
    """The script's dry-run text. Both streams joined, as above, but read in one
    go: the preview must not write anything, so it needs no log channel."""
    done = subprocess.run(_setup_argv("--dry-run"), capture_output=True, text=True,
                          timeout=_SETUP_TIMEOUT, errors="replace")
    if done.returncode != 0:
        raise RuntimeError(f"setup-mac.sh exited {done.returncode}")
    return (done.stdout or "") + (done.stderr or "")


def create_app(*, token: str | None = None, port: int | None = None) -> FastAPI:
    token = token or mint_token()
    state = AppState(token=token, port=port or 0)
    # docs_url/redoc_url/openapi_url off: they would describe every route to a
    # caller that has not proved it may read them.
    app = FastAPI(title="omnilingual-ui", docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.ui = state

    def host_ok(host: str) -> bool:
        """DNS-rebinding defence: only the two loopback spellings are legal.

        Including the port: 127.0.0.1 with no port is not the URL this server is
        reachable at, and a request that names it is not coming from the window.
        """
        return host in (f"127.0.0.1:{state.port}", f"localhost:{state.port}")

    def origin_ok(origin: str | None) -> bool:
        # Absent is allowed: not every client sends Origin. Present and foreign is
        # refused, which is what stops a page on the open web from writing keys.
        return origin is None or origin in (f"http://127.0.0.1:{state.port}",
                                            f"http://localhost:{state.port}")

    def denied(reason: str) -> JSONResponse:
        return JSONResponse({"error": reason}, status_code=403)

    @app.middleware("http")
    async def harden(request: Request, call_next):
        """The two defences that belong to every path, in one place.

        A per-route guard is one line a new route can forget; here a route is
        guarded the moment it exists, and the token exception is a named set
        instead of a decision repeated thirteen times. WebSockets skip HTTP
        middleware, so the handshake repeats these two checks itself.
        """
        host = request.headers.get("host", "")
        if not host_ok(host):
            return denied("bad host")
        origin = request.headers.get("origin")
        if not origin_ok(origin):
            return denied("bad origin")
        if request.url.path not in _UNGUARDED and \
                request.headers.get(TOKEN_HEADER) != state.token:
            return denied("bad token")
        return await call_next(request)

    @app.get("/api/health")
    async def health() -> dict:
        # Unguarded on purpose: the window has to be able to prove the server is up
        # before it has a token. It exposes no user data.
        return {"ok": True, "version": UI_VERSION,
                "arch": platform.machine(), "macos": platform.mac_ver()[0],
                "arm64": platform.machine() == "arm64",
                "python": sys.version.split()[0]}

    @app.get("/api/defaults")
    async def get_defaults(request: Request) -> dict:
        # settings.load() carries every panel field including the LiveOptions ones,
        # so the page opens on the configuration the command line would have used.
        return {**settings.load(),
                "stt_providers": list(STT_PROVIDERS),
                "mt_providers": list(MT_PROVIDERS),
                "default_stt_models": dict(DEFAULT_STT_MODELS),
                "default_mt_models": dict(DEFAULT_MT_MODELS),
                "last_output_dir": str(REPO_ROOT)}

    @app.put("/api/settings")
    async def put_settings(request: Request):
        try:
            body = await _json_object(request)
        except ConfigError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        try:
            return settings.save(body)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/keys")
    async def get_keys(request: Request) -> dict:
        return secrets.present()

    @app.put("/api/keys")
    async def put_keys(request: Request):
        try:
            body = await _json_object(request)
        except ConfigError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        try:
            for name, value in body.items():
                if value is None:
                    secrets.clear(name)
                else:
                    # Stripping and the single-line refusal belong to set_key: they
                    # are what stops a newline in a pasted value adding a second key
                    # to the file the CLI loads.
                    secrets.set_key(name, str(value))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return secrets.present()

    @app.get("/api/audio")
    async def get_audio(request: Request) -> dict:
        return _readiness(mic=True)

    @app.post("/api/audio/setup")
    async def post_audio_setup(request: Request):
        try:
            for line in audio.setup():
                state.queue.put_nowait({"type": "log", "level": "info",
                                        "message": line})
        except Exception as exc:  # noqa: BLE001 - reported to the page
            return JSONResponse({"error": str(exc)}, status_code=500)
        return _readiness(mic=False)

    @app.post("/api/audio/restart-daemon")
    async def post_restart_daemon(request: Request):
        try:
            audio.restart_daemon()
        except (OSError, subprocess.SubprocessError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        return _readiness(mic=False)

    @app.post("/api/session/start")
    async def post_start(request: Request):
        if state.busy() is not None:
            return JSONResponse(
                {"error": "a run is already in progress; stop it first"},
                status_code=409)
        try:
            body = await _json_object(request)
            built = _build(body)
            # After validation, before the thread: cli.py gates ffmpeg here too, and
            # a missing binary discovered three seconds into a run is a worse answer.
            ensure_ffmpeg()
            if built["mode"] == "live":
                _live_gate()
            runner = _start(state, built["kwargs"], built["mode"])
        except ConfigError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except (FfmpegMissingError, OSError) as exc:
            # Not a validation rejection: a precondition and the filesystem both
            # fail here, and both are the user's to fix, so both are a 400.
            return JSONResponse({"error": str(exc)}, status_code=400)
        return {"run_id": runner.run_id, "mode": built["mode"]}

    @app.post("/api/session/stop")
    async def post_stop(request: Request):
        runner = state.busy()
        if runner is None:
            return JSONResponse({"error": "no run is active"}, status_code=404)
        runner.stop()
        # The status comes back unchanged for a batch run: stop() only reaches a
        # live run, and reporting 'stopping' there would announce a stop that is
        # never coming.
        return {"run_id": runner.run_id, "status": runner.status}

    @app.get("/api/runs")
    async def get_runs(request: Request) -> dict:
        return {"runs": [
            {"run_id": r.run_id, "status": r.status,
             "out": str(r.out_path) if r.out_path else None,
             "session_dir": str(r.session_dir) if r.session_dir else None,
             "recoverable": r.recoverable}
            for r in state.runs.values()]}

    def _recover(session_dir: Path, out: Path):
        """Start a replay of a saved session's chunks.

        Shared by the route and the WebSocket's recover message so both reach the
        same recovery. Providers come from the stored panel; a recovery replays
        chunks that are already on disk, so it needs no source file and no ffmpeg.
        """
        body = settings.load()
        config, stt, translator, diarizer = build_run(body)
        return _start(state, {"session_dir": session_dir, "out": out,
                              "settings": config, "stt": stt,
                              "translator": translator, "diarizer": diarizer,
                              "english_only": bool(body.get("english_only"))},
                      "recover")

    @app.post("/api/session/recover")
    async def post_recover(request: Request):
        if state.busy() is not None:
            return JSONResponse(
                {"error": "a run is already in progress; stop it first"},
                status_code=409)
        try:
            body = await _json_object(request)
            raw = str(body.get("session_dir") or "").strip()
            if not raw:
                raise ConfigError("session_dir is required")
            session_dir = Path(raw).expanduser()
            if not session_dir.is_dir():
                raise ConfigError(f"{session_dir} is not a session dir")
            out = _output_path(body, fallback=_RECOVERED_OUT)
            runner = _recover(session_dir, out)
        except (ConfigError, OSError) as exc:
            # OSError is the output directory; ConfigError is everything else.
            return JSONResponse({"error": str(exc)}, status_code=400)
        return {"run_id": runner.run_id, "mode": "recover"}

    @app.get("/api/setup/preview")
    async def get_setup_preview(request: Request):
        try:
            return {"output": _setup_preview()}
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/setup/apply")
    async def post_setup_apply(request: Request):
        try:
            return {"output": _setup_apply(state)}
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        # The browser WebSocket API cannot set a header on the handshake, so the
        # token may arrive as a query parameter as well. Origin is checked here and
        # not in the HTTP middleware: a cross-site WebSocket is not covered by the
        # same-origin policy, which makes this the one handshake an attacker page
        # could otherwise open with no preflight at all.
        supplied = (socket.headers.get(TOKEN_HEADER)
                    or socket.query_params.get("token"))
        if not host_ok(socket.headers.get("host", "")) \
                or not origin_ok(socket.headers.get("origin")) \
                or supplied != state.token:
            await socket.close(code=1008)
            return
        # The loop that owns the socket owns the queue, and it is the loop a run's
        # thread must hand events to.
        state.loop = asyncio.get_running_loop()
        await socket.accept()
        runner = state.busy()
        await socket.send_json(
            {"type": "state", **(runner.snapshot() if runner else _idle_state())})

        async def from_page() -> None:
            while True:
                try:
                    message = await socket.receive_json()
                except ValueError:
                    # A frame that is not JSON is ignored, not fatal: one bad frame
                    # must not cost the page the rest of the meeting's transcript.
                    continue
                if not isinstance(message, dict):
                    continue
                kind = message.get("type")
                if kind == "stop":
                    live = state.busy()
                    if live is not None:
                        live.stop()
                elif kind == "recover":
                    past = state.runs.get(str(message.get("run_id") or ""))
                    if (past is not None and past.session_dir is not None
                            and past.recoverable and state.busy() is None):
                        try:
                            _recover(past.session_dir, REPO_ROOT / _RECOVERED_OUT)
                        except (ConfigError, OSError) as exc:
                            state.queue.put_nowait(
                                {"type": "log", "level": "error",
                                 "message": f"recovery refused: {exc}"})

        async def from_runs() -> None:
            while True:
                await socket.send_json(await state.queue.get())

        # Both directions at once: a page that sends nothing must still receive, and
        # a page that sends a stop must not wait for the next segment to be heard.
        reader = asyncio.create_task(from_page())
        writer = asyncio.create_task(from_runs())
        done, pending = await asyncio.wait({reader, writer},
                                           return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if pending:
            # wait, not gather: gather re-raises a cancelled child's CancelledError
            # out of the teardown, and this is a teardown. Exceptions are read below
            # either way, so nothing is left unretrieved.
            await asyncio.wait(pending)
        for task in done:
            exc = task.exception()
            # A closed window is the ordinary way this ends. Nothing here touches a
            # runner: the run is in this process, and a reconnecting page re-reads
            # it from the same registry.
            if exc is not None and not isinstance(
                    exc, (WebSocketDisconnect, RuntimeError)):
                raise exc

    @app.get("/token.js", response_class=PlainTextResponse)
    async def token_js() -> str:
        # The page needs the token and must not have it hard-coded in the HTML.
        # Same-origin and read-only, so it needs no token of its own.
        return f"window.OMNILINGUAL_TOKEN = {state.token!r};"

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        page = STATIC_DIR / "index.html"
        if page.is_file():
            return page.read_text(encoding="utf-8")
        return _PLACEHOLDER

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app
