"""The launcher: a loopback port, a daemon thread, and a token that stays out
of the output.

Nothing here needs hardware, a network, a real key or a GUI window. The tests
that start a server start it on a real ephemeral port on 127.0.0.1 and talk to it
over a real socket, because the two properties that matter — the port is served,
and the launch token never lands in anything the launcher prints — are properties
of the running process, not of any object a mock can be asked about.

The token is checked twice, on the two paths it can take. One test drives
`serve()` in-process; the other drives `-m omnilingual.ui` as a real process,
because the in-process test never runs `main()`, and `main()` is where a future
`--verbose` flag would sit — raising uvicorn's error logger after startup and
putting the handshake record back into the output without touching either switch
in `serve`. Both are also written so they cannot pass vacuously: one is the
control, which builds a server the way uvicorn builds one by default and requires
the token to BE in the captured output.

"Readiness" is identity, not liveness. `_serving` requires the app on the port to
hand back the launch token the launcher minted, so a second launcher or any other
server answering `200 {"ok": true}` is refused instead of adopted; the squatter
here answers exactly that, to every path, which is what the loser of a port race
sees. And the constraint tests here assert on closed sets — the launcher's whole
import list, the shapes that load a module without an import statement saying so,
and the modules it really loads at runtime — because a blacklist or a substring of
the source is satisfied by the alias that bypasses it. Between them the first two
catch what the launcher *names* and the last catches what it *loads*; none of them
catches an already-loaded module used through a computed attribute, and each test
says so where it applies.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import http.client
import http.server
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="the ui extra is not installed")
pytest.importorskip("uvicorn", reason="the ui extra is not installed")

from omnilingual.ui.__main__ import (  # noqa: E402
    _serving,
    build_parser,
    free_port,
    main,
    open_window,
    serve,
)
from omnilingual.ui.server import mint_token  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "omnilingual" / "ui" / "__main__.py"

# How long a test waits for a port to answer before giving up. Generous, because
# it also covers a cold first import of the app on a loaded machine; the tests
# that expect a failure pass a short one instead.
READY_TIMEOUT = 20.0

# Dropped from every child process this module starts. Nothing on the launcher's
# path reads them, but the repo's own environment carries live ones and a captured
# stream is the last place they should be able to reach.
_API_KEY_VARS = frozenset({"SARVAM_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"})


def _child_env(tmp_path) -> dict[str, str]:
    """The environment every child process here starts with.

    One helper rather than a comprehension per call site, because the scrub is the
    kind of thing that gets added at the first subprocess and forgotten at the
    second: the launcher's own child was scrubbed last round while the import-graph
    probe beside it still inherited all three live keys, and its assertion
    interpolates `result.stderr`, so a traceback that dumped the environment would
    have put them into captured pytest output. The env file is pointed at a temp
    path in the same breath, so no child can read the repo's real .env either.
    """
    env = {name: value for name, value in os.environ.items()
           if name not in _API_KEY_VARS}
    env["OMNILINGUAL_ENV_FILE"] = str(tmp_path / ".env")
    return env


# --- driving a real socket ---------------------------------------------------


def _get(port: int, path: str, *, host: str | None = None,
         timeout: float = 2.0) -> tuple[int, bytes]:
    """One plain HTTP request to the launched app. Returns (status, body)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        headers = {"Host": host} if host is not None else {}
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def _wait_until_serving(port: int, *, timeout: float = READY_TIMEOUT) -> bool:
    """Poll /api/health until the launched app answers on *port*.

    Polling the app and not just the socket is the point: another process can
    hold a port, and a bare TCP connect to it succeeds. Only the app's own
    health route proves that what answered is the server this launcher started.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, body = _get(port, "/api/health", timeout=0.5)
            # Inside the try on purpose: the body on this port is whatever else
            # holds it, so "something answered" and "what it answered" are the same
            # failure here, and a squatter's HTML error page is a poll that has not
            # succeeded yet — which is what `except ValueError` below is for. Left
            # outside, it was a JSONDecodeError out of a test helper.
            payload = json.loads(body)
        except (OSError, http.client.HTTPException, ValueError):
            time.sleep(0.05)
            continue
        # isinstance, not duck typing: the body on this port is whatever else is
        # on it, and a JSON array or string is a valid response that has no .get.
        if status == 200 and isinstance(payload, dict) and payload.get("ok") is True:
            return True
        time.sleep(0.05)
    return False


class _AnswersEverything(http.server.BaseHTTPRequestHandler):
    """Answers 200 {"ok": true} to every path, the way the app answers health."""

    def do_GET(self):  # noqa: N802 - the name http.server requires
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Silent: this listener's chatter would only pollute captured output."""


class _Squatter:
    """A loopback port held by something that answers instead of staying silent.

    The bind-failure listener below is silent — it accepts and never writes — so
    any probe at all refuses it. This one answers `200 {"ok": true}` to *every*
    path, /api/health and /token.js included, which is exactly what the loser of
    a port race sees when a second copy of the app got there first. A readiness
    check that asks only "did something answer 200 with ok in it?" adopts it.
    """

    def __init__(self):
        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                                      _AnswersEverything)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_Squatter":
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=10)


@contextlib.contextmanager
def _launcher_process(port: int, tmp_path):
    """The real launcher as `-m omnilingual.ui`, running, with its output drained.

    stdout is drained on a reader thread and stderr merged into it, so a test can
    wait for a line instead of racing the pipe, and can assert against everything
    the process wrote. The env is `_child_env`, so nothing here can read the
    repo's real .env and no live API key travels with the child.

    Yields (proc, lines); `lines` is complete only after the block exits.
    """
    env = _child_env(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, "-m", "omnilingual.ui", "--no-window", "--port", str(port)],
        cwd=str(REPO), env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines: list[str] = []
    reader = threading.Thread(target=lambda: lines.extend(proc.stdout), daemon=True)
    reader.start()
    try:
        yield proc, lines
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            proc.kill()
            proc.wait(timeout=15)
        reader.join(timeout=15)
        proc.stdout.close()


def _launch_token(port: int) -> str:
    """The token the running app minted, fetched the way the page fetches it.

    Read out of /token.js rather than handed in, so the tests use the real
    per-launch secret instead of a stand-in that might behave differently.
    """
    status, body = _get(port, "/token.js")
    assert status == 200, f"/token.js answered {status}"
    text = body.decode("utf-8")
    prefix, _, rest = text.partition("=")
    assert prefix.strip() == "window.OMNILINGUAL_TOKEN", text
    return rest.strip().rstrip(";").strip("'\"")


def _token_in_request_line(port: int, token: str, *, upgrade: bool) -> str:
    """Send `GET /ws?token=...` and return the response head.

    The browser cannot put a header on the WebSocket handshake, so the page
    carries the token in the request line — the one place the secret is exposed
    to anything that writes a log. With *upgrade* the request is the real
    handshake the page makes; without it, the same URL as an ordinary request,
    which is what an HTTP-only client (or a scanner) sends and what uvicorn's
    http protocol writes to its access log.
    """
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    lines = [f"GET /ws?token={token} HTTP/1.1", f"Host: 127.0.0.1:{port}"]
    if upgrade:
        lines += ["Upgrade: websocket", "Connection: Upgrade",
                  f"Sec-WebSocket-Key: {key}", "Sec-WebSocket-Version: 13"]
    request = "\r\n".join(lines) + "\r\n\r\n"
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(request.encode("ascii"))
        sock.shutdown(socket.SHUT_WR)
        head = b""
        while b"\r\n\r\n" not in head and len(head) < 65536:
            chunk = sock.recv(4096)
            if not chunk:
                break
            head += chunk
    return head.decode("latin-1", "replace")


def _drain_logging() -> None:
    """Wait for the server thread's log records to land in the captured output.

    The records are written synchronously by the connection's own thread, so the
    only thing being waited on is that thread finishing its write.
    """
    time.sleep(0.2)


# --- the CLI surface ---------------------------------------------------------


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
    parsed = build_parser().parse_args(["--no-window", "--port", "1234",
                                        "--host", "127.0.0.1"])
    assert parsed.no_window is True
    assert parsed.port == 1234
    assert parsed.host == "127.0.0.1"


def test_the_default_bind_is_loopback_only():
    # The default has to be loopback: the whole security story of this server is
    # that it is reachable from one machine only, and a default that named 0.0.0.0
    # would put the token-guarded API on the network with nothing else changed.
    parsed = build_parser().parse_args([])
    assert parsed.host == "127.0.0.1"
    assert parsed.port == 0


# Every module the launcher may import, at any level of the file. A launcher is
# the one file here that owns a window and a user-facing failure path, which makes
# it the tempting place for a shortcut. So the set is closed: an import that is not
# on this list is a decision to make in review, not something to remember to
# forbid afterwards.
_ALLOWED_LAUNCHER_IMPORTS = frozenset({
    "__future__",  # annotations
    "argparse",  # the CLI
    "http.client",  # the readiness probe
    "socket",  # free_port
    "sys",
    "threading",  # the serving thread
    "time",
    "uvicorn",  # the server
    "webbrowser",  # the window fallback
    "webview",  # the window
    "omnilingual.ui.server",  # the app, imported lazily by serve()
})


def _imported_modules(source: str) -> set[str]:
    """Every module an import statement in *source* names, whatever its alias.

    Walked instead of grepped, so `from omnilingual import pipeline` and
    `import subprocess as sp` come back as the same find as the plain spellings.
    A substring test over the source cannot see either of those.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.add(node.module or "")
    return found


# The callables that load a module without any import statement naming it.
# `__import__` is a builtin, so calling it needs nothing on the closed set, and
# `importlib.import_module` needs `importlib`, which is not on it either.
_DYNAMIC_IMPORT_CALLS = frozenset({"__import__", "import_module"})


def _loads_a_module_undeclared(source: str) -> set[str]:
    """Every shape in *source* that reaches a module no import statement names.

    These are the holes in a closed set built from import statements, and all three
    work from *inside a function body*, where the runtime probes never look until
    the launcher is actually serving:

    - `__import__("omnilingual.pipeline")` and `importlib.import_module(...)` are
      imports that no `import` statement in the file records.
    - `sys.modules["os"]` needs no import at all, because a launcher that is
      already running has `os`, `subprocess` and `platform` in that dictionary —
      uvicorn puts them there. So `sys.modules["os"].system(...)` runs a shell
      command with the closed set entirely satisfied, which is the shape the
      runtime audit below also cannot see, since it loads no new module.

    Reported as text, so a failure names the spelling rather than a set difference.
    """
    shapes: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else getattr(func, "id", ""))
            if name in _DYNAMIC_IMPORT_CALLS:
                first = node.args[0] if node.args else None
                target = (repr(first.value)
                          if isinstance(first, ast.Constant) else "<computed>")
                shapes.add(f"{name}({target})")
        elif isinstance(node, ast.Subscript) and _is_module_dictionary(node.value):
            shapes.add(f"{_dotted(node.value)}[...]")
    return shapes


def _is_module_dictionary(node: ast.expr) -> bool:
    """Whether *node* is `sys.modules`, another module's `modules`, or a bare one.

    Deliberately not tied to the name `sys`, so `import sys as s; s.modules[...]`
    and `from sys import modules; modules[...]` are the same find.
    """
    if isinstance(node, ast.Attribute):
        return node.attr == "modules"
    return isinstance(node, ast.Name) and node.id == "modules"


def _dotted(node: ast.expr) -> str:
    """*node* as dotted source text, or `<computed>` if it is not built from names."""
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return "<computed>"


def test_the_launcher_never_switches_the_system_output_device():
    """Nothing here may touch the machine's audio output.

    Switching the default output device is the one thing this project must never
    do — it breaks every other app on the machine the moment a run starts — and a
    launcher is the tempting place for it ("open the UI, make sure sound plays").

    Three layers, because one closed set is not enough and the previous version of
    this test claimed it was:

    1. A *closed import set* over every import statement in the file, at any depth.
       `subprocess` and `ctypes` are not on it, and the CoreAudio framework is
       reached through `ctypes`, so no spelling of `import ctypes` passes. This is
       strictly stronger than the blacklist it replaces, which duplicated an entry,
       omitted `AudioObjectSetPropertyData` — the API that actually sets the default
       output device — and named two things (`os.system`, `Popen`) that were
       unreachable without an `os` or `subprocess` import to reach them by. The API
       names are asserted too, so a file that reached one fails naming the API.
    2. No *undeclared load*: `__import__(...)`, `importlib.import_module(...)` and
       `sys.modules[...]` are rejected wherever they appear, function bodies
       included. The last one is the correction to the claim this test used to
       make. "Everything that reaches a shell needs an import" was false — a running
       launcher already has `os` and `subprocess` in `sys.modules` (uvicorn puts
       them there), so `sys.modules["os"].system("...")` ran a shell command with
       the closed set fully satisfied. It needs no import because there is nothing
       left to import.
    3. A *runtime audit* of what the launcher really loads, in
       `test_the_launchers_own_runtime_imports_stay_inside_what_it_declares`.

    What this does not prevent, deliberately: a launcher that computes its way to a
    module rather than naming it — `eval`, `getattr(sys.modules, name)`, a name
    assembled at runtime. Closing that needs an import hook or a frozen import
    table, and the threat these tests defend against is a careless or well-meaning
    future commit, not an adversary writing to the spec. The line is drawn at
    "reachable by writing it the obvious way".
    """
    source = LAUNCHER.read_text(encoding="utf-8")
    imported = _imported_modules(source)
    assert imported <= _ALLOWED_LAUNCHER_IMPORTS, (
        f"the launcher may not import {sorted(imported - _ALLOWED_LAUNCHER_IMPORTS)}; "
        "a module that can reach a shell or the CoreAudio API is not on this list, "
        "and the list is closed"
    )
    undeclared = _loads_a_module_undeclared(source)
    assert not undeclared, (
        f"the launcher loads modules without declaring them: {sorted(undeclared)}. "
        "Every module the launcher uses is named by an import statement on a closed "
        "list, so a load no statement records is a load no reviewer can see. "
        "sys.modules[...] needs no import at all — a running launcher already has "
        "os and subprocess in it, and the shell reaches CoreAudio from there"
    )
    for forbidden in ("AudioObjectSetPropertyData", "SwitchAudioSource",
                      "kAudioHardwarePropertyDefaultOutputDevice"):
        assert forbidden not in source, (
            f"{forbidden} in the launcher: the launcher must never change the "
            "system's audio output device"
        )


def test_the_launcher_does_not_import_the_pipeline(tmp_path):
    """Checked on the import graph, not on the text of the file.

    `"omnilingual.pipeline" not in source` is satisfied by
    `from omnilingual import pipeline`, which pulls in the same package without
    containing the literal it forbids — so the launcher is imported in a child
    process and `sys.modules` is asked what actually got loaded.

    uvicorn and webview are imported there too, because the launcher imports them
    inside functions and those are import sites like any other. `omnilingual.ui.
    server` is deliberately not on the list: importing the app is what serving
    *is*, and the app reaches the pipeline by design — what is forbidden is the
    launcher reaching it, on the window-owning process's own.
    """
    probe = (
        "import importlib, sys\n"
        "importlib.import_module('omnilingual.ui.__main__')\n"
        "for name in ('uvicorn', 'webview'):\n"
        "    try: importlib.import_module(name)\n"
        "    except ImportError: pass\n"
        "sys.exit('omnilingual.pipeline' in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO),
                            env=_child_env(tmp_path),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        "importing the launcher pulled in omnilingual.pipeline, so the process that "
        "owns the window loads the pipeline; the probe said "
        f"{result.stderr[-400:]!r}"
    )


# Runs the real launcher, on a real port, in a child process and reports every
# module it loaded itself. `omnilingual.ui.server` is stood in for, because the
# real app reaches the pipeline by design (`server` -> `session` -> `pipeline`) and
# that would make every audit finding ambiguous. With the app stubbed, each module
# in the delta was loaded by the launcher or by uvicorn on its behalf, which is the
# attribution that turns "the pipeline is in sys.modules" into a finding about the
# launcher.
_RUNTIME_IMPORT_PROBE = """
import http.client, importlib, json, sys, time, types

stub = types.ModuleType("omnilingual.ui.server")
async def app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-length", b"2")]})
    await send({"type": "http.response.body", "body": b"ok"})
stub.create_app = lambda **kwargs: app
sys.modules["omnilingual.ui.server"] = stub

before = set(sys.modules)
launcher = importlib.import_module("omnilingual.ui.__main__")
port = launcher.free_port()
launcher.serve(port, token="audit")

served = False
deadline = time.monotonic() + 60
while time.monotonic() < deadline:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
        try:
            conn.request("GET", "/")
            conn.getresponse().read()
        finally:
            conn.close()
        served = True
        break
    except OSError:
        time.sleep(0.05)
time.sleep(0.5)
print(json.dumps({"served": served, "gained": sorted(set(sys.modules) - before)}))
"""


def _within(name: str, package: str) -> bool:
    """Whether *name* is *package* or something inside it."""
    return name == package or name.startswith(f"{package}.")


def test_the_launchers_own_runtime_imports_stay_inside_what_it_declares(tmp_path):
    """What the launcher actually loads, as opposed to what it says it loads.

    The closed import set is source-based, so it cannot see a load that happens
    when `serve()` runs — a `__import__("omnilingual.pipeline")` inside that
    function body passes every static check in this module, and the probe above
    cannot see it either, because that probe never calls `serve()`. This one does.

    The app is stubbed precisely so the delta is attributable: `ctypes` is the only
    route to CoreAudio that this process has no legitimate use for, and it is not
    loaded by anything on the launcher's path, so finding it in the delta means the
    launcher opened that door. `omnilingual.pipeline` means the window-owning
    process loaded the pipeline, which is the constraint this asserts.

    What it cannot see is stated rather than implied: `os`, `subprocess` and
    `platform` really are in this delta, because uvicorn imports them and the
    launcher has to have uvicorn. So "did a module with shell reach get loaded" is
    not a runtime question here at all — the shell is loaded whatever the launcher
    does. What is runtime-checkable is whether the launcher *reached into* the ones
    it did not declare, and that is the static `sys.modules[...]` ban's job, next
    door. Between them: the static set and shape ban cover code, this covers the
    modules that code turned out to load.

    `uvicorn` being in the delta is the non-vacuity check. Without it an audit that
    failed to observe anything at all would pass.
    """
    result = subprocess.run([sys.executable, "-c", _RUNTIME_IMPORT_PROBE],
                            cwd=str(REPO), env=_child_env(tmp_path),
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, (
        f"the runtime import audit itself failed: {result.stderr[-600:]!r}")
    audit = json.loads(result.stdout)
    assert audit["served"], (
        "the launcher never served in the audit process, so nothing it loads while "
        f"serving was measured; it said {result.stdout[-400:]!r}")

    gained = audit["gained"]
    assert "uvicorn" in gained, (
        "the audit saw no uvicorn in the delta, so it cannot be trusted to see "
        f"anything else either; it reported {gained[:20]!r}")

    ctypes_in = [name for name in gained if _within(name, "ctypes")]
    assert not ctypes_in, (
        f"the launcher loaded {ctypes_in} at runtime. ctypes is the only route to "
        "the CoreAudio API from here and nothing on the launcher's path needs it, "
        "so this process can reach the default-output-device API and must not")

    pipeline_in = [name for name in gained if _within(name, "omnilingual.pipeline")]
    assert not pipeline_in, (
        f"the launcher loaded {pipeline_in} at runtime. The launcher must never "
        "import omnilingual.pipeline: it owns the window, and the pipeline is the "
        "app's to run")


# --- serving on a real loopback port -----------------------------------------


@pytest.fixture
def running():
    """A launched server, on a real port, always shut down afterwards.

    Yields a callable so each test can pick its own port and get back the
    (server, thread, port) triple. Extra keywords go to `serve`, for the tests
    that need the app built with a launch token they know the value of.
    """
    started: list[tuple] = []

    def _start(port: int, **kwargs):
        server, thread = serve(port, **kwargs)
        started.append((server, thread))
        return server, thread, port

    yield _start

    for server, thread in started:
        server.should_exit = True
        thread.join(timeout=15)


def test_serve_puts_the_page_and_the_api_on_the_bound_port(running):
    port = free_port()
    running(port)
    assert _wait_until_serving(port), f"nothing answered /api/health on {port}"

    status, body = _get(port, "/api/health")
    assert status == 200 and json.loads(body)["ok"] is True

    # The page, because a window pointed at the API alone is a window onto
    # nothing: the three defences of Task 7 must not lock the real page out.
    status, body = _get(port, "/")
    assert status == 200
    page = body.decode("utf-8").lower()
    assert "<!doctype html>" in page and "<title>omnilingual</title>" in page


def test_the_page_and_its_assets_load_without_a_token(running):
    # The browser cannot attach a header to <script src>, so the page's own files
    # have to answer unguarded; if this regressed the window would come up blank
    # with a 403 no script could read. Asserted here because the launcher is what
    # puts that page in front of a user.
    port = free_port()
    running(port)
    assert _wait_until_serving(port)
    status, body = _get(port, "/token.js")
    assert status == 200 and b"OMNILINGUAL_TOKEN" in body
    status, body = _get(port, "/static/app.js")
    assert status == 200 and body


def test_the_launched_server_only_accepts_requests_naming_its_own_port(running):
    """serve() must tell the app which port it landed on.

    A port-less app cannot compare a Host authority against a fixed number, so
    it accepts any loopback port and its Origin check falls back to comparing the
    request's Host against its Origin. Passing the port in is what makes both
    checks strict, and the only place it can come from is the launcher — the
    server never learns the port on its own.
    """
    port = free_port()
    running(port)
    assert _wait_until_serving(port)

    status, _ = _get(port, "/api/health", host=f"localhost:{port + 1}")
    assert status == 403, "a Host naming another port must be refused"

    status, _ = _get(port, "/api/health", host=f"127.0.0.1:{port}")
    assert status == 200


def test_readiness_is_this_launchs_own_app_and_not_just_a_200_with_ok(running):
    """`200 {"ok": true}` is not proof of anything; the launch token is.

    One live server, built with a token this test knows, asked twice: with its own
    token the readiness check passes, with a different one it fails. The second
    half is what gives the first half teeth — a check that returned True for
    anything answering on the port would satisfy the first assertion alone.
    """
    token = mint_token()
    server, thread, port = running(free_port(), token=token)
    assert _wait_until_serving(port), "the launched server never came up"

    assert _serving("127.0.0.1", port, token=token, thread=thread,
                    timeout=READY_TIMEOUT) is True, (
        "the app built with our token must be recognised as ours")
    assert _serving("127.0.0.1", port, token=mint_token(), thread=thread,
                    timeout=1.0) is False, (
        "an app that hands back a different token is somebody else's server, "
        "however healthy it answers")


def test_a_dead_serving_thread_fails_the_readiness_check_at_once():
    """The bind happens inside the serving thread, so its death is already the answer.

    Waiting out the whole startup timeout instead turns a port collision into
    twenty seconds of silence before saying anything, when the cause was decided
    within a fifth of a second.
    """
    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join()

    started = time.monotonic()
    assert _serving("127.0.0.1", free_port(), token="irrelevant", thread=thread,
                    timeout=READY_TIMEOUT) is False
    assert time.monotonic() - started < 1.0, (
        "a serving thread that has died must end the readiness wait, not be "
        "waited out for the full startup timeout"
    )


# --- the token must not reach anything the launcher writes -------------------


@pytest.fixture
def uvicorn_logging():
    """Restore uvicorn's loggers, which a started server reconfigures globally.

    Config.configure_logging() rewrites handlers and levels on the shared
    `uvicorn` logger tree, so a test that starts a server would otherwise change
    what every later test sees.
    """
    names = ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi")
    saved = {name: (logging.getLogger(name).handlers,
                    logging.getLogger(name).level,
                    logging.getLogger(name).propagate)
             for name in names}
    yield
    for name, (handlers, level, propagate) in saved.items():
        logger = logging.getLogger(name)
        logger.handlers = handlers
        logger.level = level
        logger.propagate = propagate


def test_the_token_does_reach_uvicorns_access_log_when_it_is_left_on(
        uvicorn_logging, capfd):
    """The control: the leak this task is about is real, and this harness sees it.

    A server built the way uvicorn builds one by default, asked the same question
    at the same URL, and the token comes back in the captured output. Without
    this, the launcher's own test below could be passing because the capture is
    broken rather than because the launcher silenced the log.
    """
    import uvicorn

    from omnilingual.ui.server import create_app

    port = free_port()
    config = uvicorn.Config(create_app(port=port), host="127.0.0.1", port=port)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        assert _wait_until_serving(port), "the control server never came up"
        token = _launch_token(port)
        _token_in_request_line(port, token, upgrade=False)
        _drain_logging()
        captured = capfd.readouterr()
    finally:
        server.should_exit = True
        thread.join(timeout=15)

    assert token in captured.out + captured.err, (
        "uvicorn's default access log should have written the request line, "
        "token and all — if this fails the harness cannot see the leak it is "
        "the control for"
    )


def test_the_launch_token_never_reaches_the_launchers_output(uvicorn_logging, capfd):
    """The launch token must not appear in anything the launcher writes.

    The page cannot send the token as a header on the WebSocket handshake, so it
    travels in the request line — and uvicorn writes request lines into its logs.
    Two different loggers are involved, so both requests below matter: an
    ordinary request (logged by the http protocol on `uvicorn.access`, which
    `access_log=False` silences whatever the log level is) and the real handshake
    (logged by the websocket protocol on `uvicorn.error` at INFO, which is why
    the launcher also lowers the log level).
    """
    port = free_port()
    server, thread = serve(port)
    try:
        assert _wait_until_serving(port), "the launched server never came up"
        token = _launch_token(port)

        # Take the access logger off its leash so this asserts what the launcher
        # configured, not what the log level happens to filter out. Without
        # access_log=False the handler is still attached and the record is
        # emitted here; with it, the handler is gone and nothing is emitted.
        access = logging.getLogger("uvicorn.access")
        previous_level = access.level
        access.setLevel(logging.DEBUG)
        try:
            _token_in_request_line(port, token, upgrade=False)
            head = _token_in_request_line(port, token, upgrade=True)
            _drain_logging()
        finally:
            access.setLevel(previous_level)

        assert head.startswith("HTTP/1.1 101"), (
            f"the real handshake should have been accepted, got {head[:40]!r}")
        captured = capfd.readouterr()
    finally:
        server.should_exit = True
        thread.join(timeout=15)

    written = captured.out + captured.err
    assert token not in written, (
        "the launch token reached the launcher's output. Both switches in serve() "
        "are load-bearing and the repair is to keep them, not to loosen them: "
        "access_log=False is what stops the http protocol writing request lines at "
        "all, and log_level='warning' is what keeps the websocket handshake record "
        "off uvicorn.error. Do not add a verbosity flag that lowers either."
    )


# --- the window is a convenience, never a dependency -------------------------


@pytest.fixture
def no_window_module(monkeypatch):
    """Make `import webview` fail the way a machine without the extra does.

    None in sys.modules raises ImportError on import, which is the same code path
    a missing pywebview takes, so the fallback is exercised without uninstalling
    anything.
    """
    monkeypatch.setitem(sys.modules, "webview", None)


def test_no_window_opens_neither_a_window_nor_the_browser(monkeypatch):
    """`--no-window` means one thing: nothing is opened.

    The contrast is what gives this teeth. The same recorder is called both ways,
    and the second call has to land in it — so an `open_window` that never opened
    anything, which the empty-only version of this test could not tell from a
    correct one, fails here.
    """
    opened: list[str] = []

    class Recorder:
        def create_window(self, title, url, **kwargs):
            opened.append(url)

        def start(self, **kwargs):
            pass

    monkeypatch.setitem(sys.modules, "webview", Recorder())
    monkeypatch.setattr("webbrowser.open", opened.append)

    open_window("http://127.0.0.1:9999/", no_window=True)
    assert opened == [], "--no-window must not open a window or the browser"

    open_window("http://127.0.0.1:9999/")
    assert opened == ["http://127.0.0.1:9999/"], (
        "without --no-window the window is opened, so the assertion above is "
        "about the flag and not about a launcher that opens nothing at all"
    )


def test_open_window_falls_back_to_the_browser_without_pywebview(no_window_module,
                                                                 monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    open_window("http://127.0.0.1:9999/")
    assert opened == ["http://127.0.0.1:9999/"]


def test_open_window_falls_back_when_the_window_cannot_open(monkeypatch):
    """A GUI-less session raises from pywebview rather than failing the launch.

    Over SSH, in CI, or with no window server, `webview.create_window`/`start`
    raise. The UI is reachable in a browser either way, so the launcher must not
    die over it.
    """
    class NoGuiSession:
        def create_window(self, *args, **kwargs):
            raise RuntimeError("no GUI session available")

        def start(self, *args, **kwargs):  # pragma: no cover - never reached
            raise AssertionError("start() must not be reached")

    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    monkeypatch.setitem(sys.modules, "webview", NoGuiSession())
    open_window("http://127.0.0.1:9999/")
    assert opened == ["http://127.0.0.1:9999/"]


# --- failing loudly instead of hanging ---------------------------------------


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_main_reports_a_bind_failure_instead_of_waiting_forever(capfd):
    """A port already in use must be an error, not a hang and not a false success.

    The port is held by a socket that accepts and says nothing, which is what a
    bind collision looks like from the outside. A launcher that only checked
    "can I connect?" would see this listener, decide the server was up, and open
    a window onto somebody else's port.

    The thread warning is uvicorn's own way of failing: it raises SystemExit on
    the serving thread, which is what is being provoked here.
    """
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        code = main(["--no-window", "--port", str(port)], startup_timeout=1.0)
        captured = capfd.readouterr()
    finally:
        blocker.close()

    assert code == 1, "a bind failure must be a non-zero exit"
    assert f":{port}" in captured.err, (
        f"the error must name the port it could not bind, got {captured.err!r}")


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_main_refuses_a_port_held_by_a_server_that_answers(capfd):
    """A stranger's `200 {"ok": true}` is not our server, and the launcher says so.

    The end-to-end half of the identity check, and the case the silent listener
    above cannot reach: a real HTTP server holds the port and answers every path
    exactly as the app does. A readiness check that only asks "did something
    answer 200 with ok in it?" passes, prints this stranger's URL as its own,
    exits 0, and in the windowed path opens a second window onto the stranger's
    app — where the app's own one-run-at-a-time guard then answers 409.
    """
    with _Squatter() as squatter:
        code = main(["--no-window", "--port", str(squatter.port)],
                    startup_timeout=5.0)
        port = squatter.port
    captured = capfd.readouterr()

    assert code == 1, (
        "a port held by somebody else's server must be refused, not reported as "
        f"ours; the launcher exited {code} having said {captured.out!r}"
    )
    assert captured.out == "", "a refused launch must not print a URL"
    assert f":{port}" in captured.err, (
        f"the error must name the port it could not use, got {captured.err!r}")


def test_the_launcher_serves_until_it_is_stopped(tmp_path):
    """`--no-window` serves and prints the URL; it does not print and exit.

    Started as a subprocess because this is the only way to see what the launcher
    does with a real terminal-less process: no window, no GUI session, and a
    parent that can kill it. The child's output is drained on a thread so the URL
    can be waited for instead of raced — the server can answer /api/health in the
    microseconds before main() reaches its print, and a test that terminated on
    the health answer would be asserting on that race rather than on the launcher.

    Liveness is then sampled for two whole seconds, not once. The divergence this
    forbids is print-and-exit, and that happens within milliseconds of the URL
    appearing: a single poll at one instant can still catch the process alive and
    pass a launcher that printed and died.
    """
    port = free_port()
    url = f"http://127.0.0.1:{port}/"
    with _launcher_process(port, tmp_path) as (proc, lines):
        served = _wait_until_serving(port, timeout=READY_TIMEOUT)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not any(url in line for line in lines):
            time.sleep(0.05)
        announced = any(url in line for line in lines)

        exited_with = None
        until = time.monotonic() + 2.0
        while time.monotonic() < until:
            if proc.poll() is not None:
                exited_with = proc.poll()
                break
            time.sleep(0.05)
        output = "".join(lines)

    assert served, f"the launcher never served /api/health; output was {output!r}"
    assert announced, f"the launcher never printed its URL; output was {output!r}"
    assert exited_with is None, (
        f"--no-window printed its URL and exited {exited_with}; it has to keep "
        f"serving until it is stopped. Output was {output!r}"
    )


def test_the_launch_token_never_reaches_the_output_of_the_real_launcher(tmp_path):
    """The same guarantee, on the path a user actually runs: `-m omnilingual.ui`.

    The in-process test above calls `serve()` directly, so `main()` never runs —
    and `main()` is where a future `--verbose` flag would sit, putting the
    handshake record back by raising uvicorn's error logger after startup without
    touching either switch in `serve`. So this drives the launcher as a real
    process: it reads the minted token out of the running app, sends the real
    WebSocket handshake with it in the request line, and requires the token to be
    in neither stream the process wrote.
    """
    port = free_port()
    with _launcher_process(port, tmp_path) as (_proc, lines):
        assert _wait_until_serving(port, timeout=READY_TIMEOUT), (
            f"the launcher never served; output was {''.join(lines)!r}")
        token = _launch_token(port)
        head = _token_in_request_line(port, token, upgrade=True)
        _drain_logging()
    output = "".join(lines)

    assert head.startswith("HTTP/1.1 101"), (
        "the real handshake should have been accepted, so the token really did "
        f"reach uvicorn's websocket logger; got {head[:60]!r}"
    )
    assert token not in output, (
        "the launch token reached the output of the real launcher process. Both "
        "switches in serve() are load-bearing and the repair is to keep them: "
        "access_log=False, and log_level='warning' — do not let anything in main() "
        f"lower uvicorn.error's level afterwards. Output was {output!r}"
    )


def test_the_launcher_refuses_a_non_loopback_bind(capsys):
    """--host may name loopback and nothing else.

    The app's three defences — a per-launch token, a Host check against loopback,
    an Origin check — all assume one machine. On a routable address none of them
    hold, so the launcher refuses rather than serving the API to the network
    because a flag asked it to.
    """
    for host in ("0.0.0.0", "192.168.1.10", "::"):
        assert main(["--host", host, "--port", "0", "--no-window"]) == 2
    captured = capsys.readouterr()
    assert "loopback" in captured.err
    assert captured.out == "", "a refused launch must not print a URL"
