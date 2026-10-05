"""Launch the UI: serve on a free loopback port and open a window.

The window is a convenience, never a dependency. If pywebview is unavailable or
refuses to open, the app falls back to the system browser and everything still
works, because all the logic lives in the server.

The launcher owns three things the server cannot: the port, the launch token,
and what is written to the terminal. The port is picked here because uvicorn only
reports one after it has bound, and the app needs it *before* that to make its
Host and Origin checks strict — so the launcher binds an ephemeral loopback port
itself and hands the number to both. The token is minted here for the same kind
of reason: it is also how the launcher proves the answer on the port came from
*its* server and not from whatever else got there first (see `_serving`). The
third is the terminal, because the page cannot put a header on the WebSocket
handshake, so the token travels in the request line, and request lines are exactly
what server logs are made of. See `serve` for the logging that keeps it out of
them, and `_serving` for what the token buys.
"""

from __future__ import annotations

import argparse
import http.client
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
                        help="bind address; loopback only, anything else is refused")
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


def serve(port: int, *, host: str = "127.0.0.1",
          token: str | None = None) -> tuple[object, threading.Thread]:
    """Start uvicorn on a daemon thread. Returns (server, thread).

    *token* is the launch token to build the app with. `main` mints one and passes
    it in, because a caller that could not name the token could not check whether
    the thing answering the port is its own server — see `_serving`. Left None, the
    app mints one for itself, which is right for a library caller and wrong for
    the launcher.

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

    # The port and the token both go to the app as well as to uvicorn. The port is
    # what lets the app compare a Host authority against one fixed number instead
    # of accepting any loopback port, and what lets it refuse an Origin on a
    # different port. Passing None here turns both checks loose at exactly the
    # moment they start mattering: uvicorn only learns the port after it binds, and
    # the launcher is the only thing that knows it in advance. The token is the
    # launcher's handle on the app it started, for `_serving` to check against.
    config = uvicorn.Config(create_app(port=port, token=token), host=host, port=port,
                            log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True,
                              name="omnilingual-ui-server")
    thread.start()
    return server, thread


def _serving(host: str, port: int, *, token: str,
             thread: threading.Thread, timeout: float) -> bool:
    """Whether the app *this launcher started* is answering on *port*.

    Identity, not liveness. A `200` with `{"ok": true}` proves only that something
    speaks HTTP there, and something else can: a second launcher holding the port
    answers exactly that, so a liveness check calls a stranger's server its own,
    prints the winner's URL, and in the windowed path opens a second window onto
    the winner's app. `free_port` binds and releases, so the gap between its
    release and uvicorn's bind is a real window, not a theoretical one.

    So the proof is a value only this launch could have: `main` mints the launch
    token, hands the same one to `create_app`, and `/token.js` — the route the page
    reads it from — hands it back. An app this launcher did not start was given a
    different token and cannot answer with this one.

    The serving thread is the other half, and it is what makes the answer fast.
    The bind happens inside that thread, so a collision is already decided by the
    time this sees the thread gone; returning then turns twenty seconds of silence
    into an immediate error naming the cause.
    """
    deadline = time.monotonic() + timeout
    while thread.is_alive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if _echoes(host, port, token, timeout=min(1.0, remaining)):
            return True
        time.sleep(0.05)
    return False


def _echoes(host: str, port: int, token: str, *, timeout: float) -> bool:
    """Whether the server on *port* hands back *token* — ours, or not."""
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        try:
            conn.request("GET", "/token.js")
            response = conn.getresponse()
            body = response.read()
        finally:
            conn.close()
    except (OSError, http.client.HTTPException):
        return False
    if response.status != 200:
        return False
    # The exact bytes the route writes, trailing semicolon included, so a
    # squatter answering this path with anything of its own fails here.
    return body.decode("utf-8", "replace") == f"window.OMNILINGUAL_TOKEN = {token!r};"


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

    Sliced rather than one blocking join, so the wait polls in bounded steps. The
    thread's own death is not a reason for this loop to exist: `main` has proved
    the server answers before it gets here, and a thread that died instead has
    already been reported as an error.
    """
    while thread.is_alive():
        thread.join(timeout=0.5)


def main(argv: list[str] | None = None, *,
         startup_timeout: float = _STARTUP_TIMEOUT) -> int:
    """Run the launcher. Exit 2 for a wrong invocation, 1 for one the machine refused.

    The split is Click's: a flag that cannot be honoured is a usage error, and a
    port the OS will not give us is a failure no argument could have prevented.
    """
    args = build_parser().parse_args(argv)
    if args.host not in _LOOPBACK:
        # The three defences assume one machine: a token another process must
        # guess, a Host check against loopback, and an Origin check. On a routable
        # address none of them hold, so this is refused rather than warned about.
        print(f"--host must be loopback ({' or '.join(_LOOPBACK)}), got {args.host!r}",
              file=sys.stderr)
        return 2

    # Minted here rather than left to the app, because the launcher has to be able
    # to name the token to check that the thing on the port is its own server — and
    # imported this late so that --help and a refused --host cost no app import.
    from omnilingual.ui.server import mint_token

    token = mint_token()
    port = args.port or free_port()
    server, thread = serve(port, host=args.host, token=token)
    url = f"http://{args.host}:{port}/"

    # uvicorn binds inside its own thread, so wait for the app to answer rather
    # than assuming the thread start means the port is live.
    if not _serving(args.host, port, token=token, thread=thread,
                    timeout=startup_timeout):
        if thread.is_alive():
            print(f"server did not come up on {url}", file=sys.stderr)
        else:
            # uvicorn has already said why, immediately above this line: address
            # already in use, a port below 1024, a host that is not this machine.
            print(f"could not bind to {url}", file=sys.stderr)
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
