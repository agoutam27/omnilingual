"""Launch the UI: serve on a free loopback port and open a window.

The window is a convenience, never a dependency. If pywebview is unavailable or
refuses to open, the app falls back to the system browser and everything still
works, because all the logic lives in the server.

The launcher owns two things the server cannot: the port, and what is written to
the terminal. The port is picked here because uvicorn only reports one after it
has bound, and the app needs it *before* that to make its Host and Origin checks
strict — so the launcher binds an ephemeral loopback port itself and hands the
number to both. The second is the launch token: the page cannot put a header on
the WebSocket handshake, so the token travels in the request line, and request
lines are exactly what server logs are made of. See `serve` for the logging that
keeps it out of them.
"""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import sys
import threading
import time
import webbrowser

# How long the launcher waits for its own app to answer before it gives up and
# says so. Long enough for a cold first import on a loaded machine, short enough
# that a bind collision is an error the user sees rather than a hang.
_STARTUP_TIMEOUT = 20.0

# The only bind addresses offered. Both spellings are the same machine, and both
# are the ones the app's Host and Origin checks accept.
_LOOPBACK = ("127.0.0.1", "localhost")


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


def serve(port: int, *, host: str = "127.0.0.1") -> tuple[object, threading.Thread]:
    """Start uvicorn on a daemon thread. Returns (server, thread).

    Two of these settings are load-bearing rather than taste, and both are about
    the launch token. The page cannot send a header on the WebSocket handshake,
    so it carries the token in the request line, and request lines are what
    uvicorn's logs are made of.

    `access_log=False` is not redundant with `log_level`. uvicorn's http protocol
    writes one line per request — request line, query string and all — to the
    `uvicorn.access` logger, and its handlers come off only when `access_log` is
    False (uvicorn/config.py: `if self.access_log is False:
    logging.getLogger("uvicorn.access").handlers = []`). A log level is a
    different mechanism: it filters records and leaves the handler attached, so
    any later `setLevel` in the process re-opens the leak. Switching the access
    log off removes the capability instead of muting it.

    `log_level="warning"` is needed as well, because the token's other route is
    not the access log. The websocket protocol logs the handshake on
    `uvicorn.error` at INFO — `'%s - "WebSocket %s" [accepted]'`, built from
    get_path_with_query_string, so the query string, and the token, are in it —
    and the log level is what keeps that record out. Neither setting alone covers
    both paths.
    """
    import uvicorn

    from omnilingual.ui.server import create_app

    # The port goes to the app as well as to uvicorn. It is what lets the app
    # compare a Host authority against one fixed number instead of accepting any
    # loopback port, and what lets it refuse an Origin on a different port.
    # Passing None here turns both checks loose at exactly the moment they start
    # mattering: uvicorn only learns the port after it binds, and the launcher is
    # the only thing that knows it in advance.
    config = uvicorn.Config(create_app(port=port), host=host, port=port,
                            log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True,
                              name="omnilingual-ui-server")
    thread.start()
    return server, thread


def _serving(host: str, port: int, *, timeout: float) -> bool:
    """Whether the app this launcher started is answering on *port*.

    The app's own health route, not a bare TCP connect: a port already in use by
    something else accepts connections perfectly well, and connecting to *that*
    would be reported as this launcher's server being up — followed by a window
    opened onto whatever is really there. Only our own route settles it.
    """
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            conn = http.client.HTTPConnection(host, port, timeout=min(1.0, remaining))
            try:
                conn.request("GET", "/api/health")
                response = conn.getresponse()
                body = response.read()
            finally:
                conn.close()
            if response.status == 200 and json.loads(body).get("ok") is True:
                return True
        except (OSError, ValueError, http.client.HTTPException):
            pass
        time.sleep(0.05)


def _stop(server, thread: threading.Thread) -> None:
    """Ask the serving thread to finish, then wait for it."""
    server.should_exit = True
    thread.join(timeout=10)
    if thread.is_alive():  # pragma: no cover - uvicorn that will not stop
        print("the UI server thread did not stop", file=sys.stderr)


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


def _serve_until_interrupted(thread: threading.Thread) -> None:
    """Block in the foreground until Ctrl-C, for --no-window.

    Joining in slices rather than once so the wait stays interruptible, and so
    the thread's death (a bind error, say) ends the wait on its own.
    """
    while thread.is_alive():
        thread.join(timeout=0.5)


def main(argv: list[str] | None = None, *,
         startup_timeout: float = _STARTUP_TIMEOUT) -> int:
    args = build_parser().parse_args(argv)
    if args.host not in _LOOPBACK:
        # The three defences assume one machine: a token another process must
        # guess, a Host check against loopback, and an Origin check. On a routable
        # address none of them hold, so this is refused rather than warned about.
        print(f"--host must be loopback ({' or '.join(_LOOPBACK)}), got {args.host!r}",
              file=sys.stderr)
        return 2
    port = args.port or free_port()
    server, thread = serve(port, host=args.host)
    url = f"http://{args.host}:{port}/"

    # uvicorn binds inside its own thread, so wait for the app to answer rather
    # than assuming the thread start means the port is live.
    if not _serving(args.host, port, timeout=startup_timeout):
        print(f"server did not come up on {url}", file=sys.stderr)
        _stop(server, thread)
        return 1

    # flush: piped and redirected stdout is block-buffered, and the one thing
    # this line says is the URL someone is about to need in another terminal.
    print(f"omnilingual UI on {url}", flush=True)

    try:
        if args.no_window:
            _serve_until_interrupted(thread)
        else:
            open_window(url, no_window=args.no_window)
    except KeyboardInterrupt:
        pass
    finally:
        _stop(server, thread)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())