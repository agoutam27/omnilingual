"""The launcher: a loopback port, a daemon thread, and a token that stays out
of the output.

Nothing here needs hardware, a network, a real key or a GUI window. The tests
that start a server start it on a real ephemeral port on 127.0.0.1 and talk to it
over a real socket, because the two properties that matter — the port is served,
and the launch token never lands in anything the launcher prints — are properties
of the running process, not of any object a mock can be asked about.

The token tests are written so they cannot pass vacuously. One of them is the
control: it builds a server the way uvicorn builds one by default, drives the
same request at it, and requires the token to BE in the captured output. The
launcher's own test then requires the token to be absent from the same request's
output. So "absent" means "the launcher configured it away", and a launcher that
left uvicorn's default in place fails rather than quietly satisfying the
assertion.
"""

from __future__ import annotations

import base64
import http.client
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
    build_parser,
    free_port,
    main,
    open_window,
    serve,
)

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "omnilingual" / "ui" / "__main__.py"

# How long a test waits for a port to answer before giving up. Generous, because
# it also covers a cold first import of the app on a loaded machine; the tests
# that expect a failure pass a short one instead.
READY_TIMEOUT = 20.0


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
        except (OSError, http.client.HTTPException):
            time.sleep(0.05)
            continue
        if status == 200 and json.loads(body).get("ok") is True:
            return True
        time.sleep(0.05)
    return False


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


def test_the_launcher_never_switches_the_system_output_device():
    """Nothing here may touch the machine's audio output.

    Switching the default output device is the one thing this project must never
    do — it breaks every other app on the machine the moment a run starts. A
    launcher is the tempting place for it (open the UI, "make sure sound plays"),
    so the prohibition is pinned on the launcher's own source.
    """
    source = LAUNCHER.read_text(encoding="utf-8")
    for forbidden in ("SwitchAudioSource", "kAudioHardwarePropertyDefaultOutputDevice",
                      "SwitchAudioSource", "os.system", "Popen"):
        assert forbidden not in source, (
            f"{forbidden} in the launcher: the launcher must never change the "
            "system's audio output device"
        )


def test_the_launcher_does_not_import_the_pipeline():
    # The web layer reaches a run only through omnilingual.ui.session; importing
    # the pipeline here would pull a heavy package into the process that owns the
    # window. Mirrors the server.py rule in tests/test_ui_packaging.py.
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "omnilingual.pipeline" not in source


# --- serving on a real loopback port -----------------------------------------


@pytest.fixture
def running():
    """A launched server, on a real port, always shut down afterwards.

    Yields a callable so each test can pick its own port and get back the
    (server, thread, port) triple.
    """
    started: list[tuple] = []

    def _start(port: int):
        server, thread = serve(port)
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
        "the launch token reached the launcher's output — remove access_log=False "
        "or lower log_level and this is what happens"
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


def test_open_window_does_nothing_when_the_window_is_refused(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    monkeypatch.setitem(sys.modules, "webview", object())
    open_window("http://127.0.0.1:9999/", no_window=True)
    assert opened == [], "--no-window must not open anything at all"


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


def test_the_launcher_serves_until_it_is_stopped(tmp_path):
    """`--no-window` serves and prints the URL; it does not print and exit.

    Started as a subprocess because this is the only way to see what the launcher
    does with a real terminal-less process: no window, no GUI session, and a
    parent that can kill it. The child's output is drained on a thread so the URL
    can be waited for instead of raced — the server can answer /api/health in the
    microseconds before main() reaches its print, and a test that terminated on
    the health answer would be asserting on that race rather than on the launcher.
    """
    port = free_port()
    env = {**os.environ,
           # A temp env file so nothing here can read the repo's real .env, which
           # holds live API keys.
           "OMNILINGUAL_ENV_FILE": str(tmp_path / ".env")}
    proc = subprocess.Popen(
        [sys.executable, "-m", "omnilingual.ui", "--no-window", "--port", str(port)],
        cwd=str(REPO), env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines: list[str] = []
    reader = threading.Thread(target=lambda: lines.extend(proc.stdout), daemon=True)
    reader.start()
    url = f"http://127.0.0.1:{port}/"
    try:
        served = _wait_until_serving(port, timeout=READY_TIMEOUT)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not any(url in line for line in lines):
            time.sleep(0.05)
        announced = any(url in line for line in lines)
        alive = proc.poll() is None
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            proc.kill()
            proc.wait(timeout=15)
        reader.join(timeout=15)
        proc.stdout.close()

    output = "".join(lines)
    assert served, f"the launcher never served /api/health; output was {output!r}"
    assert announced, f"the launcher never printed its URL; output was {output!r}"
    assert alive, "the launcher exited instead of keeping the server up"


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