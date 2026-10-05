"""The ASGI server: routes, the event stream, and the loopback guards.

Every test here runs with no audio hardware, no network egress and no real API
key. The runners are replaced by FakeRunner wherever a request would otherwise
spawn a thread, so what is exercised is the HTTP layer: the guards, the payload
that reaches session.build_run, and the events that reach the page.

The credential gate is NOT stubbed. A start request builds real providers from
real Settings through the one function that is allowed to do so, which is the
only way a regression in that gate could be caught here.
"""

from __future__ import annotations

import asyncio
import io
import pathlib
import subprocess
import threading
import time

import pytest
from fastapi.testclient import TestClient

from omnilingual.ui import server
from omnilingual.ui.server import create_app, mint_token

HOST = "127.0.0.1:5599"
PORT = 5599
SCRIPT = server.SETUP_SCRIPT

# Every route that must answer 403 without the token. A new route has to be added
# here, and the table is checked against the app itself, so a route cannot be
# added and left unguarded.
RESERVED = [
    ("GET", "/api/defaults"),
    ("PUT", "/api/settings"),
    ("GET", "/api/keys"),
    ("PUT", "/api/keys"),
    ("GET", "/api/audio"),
    ("POST", "/api/audio/setup"),
    ("POST", "/api/audio/restart-daemon"),
    ("POST", "/api/session/start"),
    ("POST", "/api/session/stop"),
    ("GET", "/api/runs"),
    ("POST", "/api/session/recover"),
    ("GET", "/api/setup/preview"),
    ("POST", "/api/setup/apply"),
]

FAKE_KEY = "test-key-not-a-credential"
FAKE_SECRET = "sk-super-secret-value"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """An isolated .env and settings store, and a launch environment with no keys.

    run_env() layers the .env over os.environ, so a developer who exported
    SARVAM_API_KEY before running pytest would otherwise make every
    "a missing key is rejected" test pass or fail by accident.
    """
    for name in ("SARVAM_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    dotenv = tmp_path / ".env"
    dotenv.write_text("", encoding="utf-8")
    monkeypatch.setenv("OMNILINGUAL_ENV_FILE", str(dotenv))
    return dotenv


@pytest.fixture
def api(env):
    token = mint_token()
    return TestClient(create_app(token=token, port=PORT)), token


def _ok(token, **extra):
    return {"X-Omnilingual-Token": token, "Host": HOST, **extra}


def _start_body(**over):
    body = {"mode": "live", "stt": "sarvam", "mt": "mayura"}
    body.update(over)
    return body


def _state(client):
    return client.app.state.ui


def _logs(client):
    return [m["message"] for m in list(_state(client).queue._queue)
            if m["type"] == "log"]


def _wait_for_logs(client, count, timeout=10.0):
    """The log messages a detached operation has produced, waiting for `count`.

    Long operations run on a worker thread now, so their output arrives after the
    response does. Polling is what keeps the assertions deterministic without
    putting a join handle in the production module purely for the tests.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len(_logs(client)) < count:
        time.sleep(0.01)
    return _logs(client)


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.01)
    return predicate()


def _readiness(**over):
    from omnilingual.ui import audio

    fields = dict(device=True, blackhole=True, ffmpeg=True, ffprobe=True,
                  mic_authorized=True, output="Omnilingual", detail=[])
    fields.update(over)
    return audio.Readiness(**fields)


class FakeRunner:
    """A SessionRunner that records what it was asked to do and starts no thread."""

    def __init__(self, *, run_id, queue, loop=None):
        self.run_id = run_id
        self.queue = queue
        self.loop = loop
        self.calls: list[tuple[str, dict]] = []
        self.status = "running"
        self._out = None
        self._session_dir = None

    def _record(self, kind, kwargs):
        self.calls.append((kind, kwargs))
        self._out = kwargs.get("out") or kwargs["opts"].out

    def start_live(self, **kwargs):
        self._record("live", kwargs)

    def start_recording(self, **kwargs):
        self._record("recording", kwargs)

    def recover(self, **kwargs):
        self._record("recover", kwargs)
        self._session_dir = kwargs["session_dir"]

    def stop(self):
        """Only a live run can be stopped, so only a live run reports stopping."""
        if self.calls and self.calls[0][0] == "live":
            self.status = "stopping"

    @property
    def out_path(self):
        return self._out

    @property
    def session_dir(self):
        return self._session_dir

    @property
    def recoverable(self):
        return self._session_dir is not None

    def snapshot(self):
        return {"run_id": self.run_id, "mode": "live", "status": self.status,
                "elapsed_s": 0.0, "cost_inr": 0.0, "cost_cap": None,
                "segments": 0, "dropped": 0, "out": None, "session_dir": None,
                "error": None}


@pytest.fixture
def fake_runs(api, monkeypatch):
    """Swap in FakeRunner and skip the two hardware gates: no thread, no capture.

    audio.probe is stubbed ready as well as ensure_ffmpeg: a live start refuses to
    run when capture cannot work (§9), and that check compiles the device helper
    and records one second of audio. A test that wants to exercise the check
    itself stubs probe with the failure it is asserting on.
    """
    from omnilingual.ui import audio

    monkeypatch.setattr(server, "SessionRunner", FakeRunner)
    monkeypatch.setattr(server, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(audio, "probe", lambda *a, **k: _readiness())
    return api


def _runner(client, run_id=None):
    runs = _state(client).runs
    if run_id is None:
        assert len(runs) == 1, f"expected exactly one run, got {list(runs)}"
        return next(iter(runs.values()))
    return runs[run_id]


# --- health and environment ------------------------------------------------


def test_health_needs_no_token_and_reports_the_environment(api):
    client, _ = api
    res = client.get("/api/health", headers={"Host": HOST})
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert set(body) >= {"ok", "version", "arch", "macos", "arm64", "python"}


def test_health_reveals_no_user_data(api, env):
    """The one unguarded route must answer only about the process."""
    env.write_text(f"SARVAM_API_KEY={FAKE_SECRET}\n", encoding="utf-8")
    client, _ = api
    res = client.get("/api/health", headers={"Host": HOST})
    assert FAKE_SECRET not in res.text
    assert "/api" not in res.text


# --- the three loopback defences ------------------------------------------


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
    res = client.get("/api/defaults", headers=_ok(token, Origin="https://evil.example"))
    assert res.status_code == 403


def test_host_header_without_a_port_is_rejected(api):
    """127.0.0.1 bare is not the URL the server is reachable at, so it is refused."""
    client, token = api
    res = client.get("/api/defaults",
                     headers={"X-Omnilingual-Token": token, "Host": "127.0.0.1"})
    assert res.status_code == 403


def test_our_own_origin_is_allowed(api):
    """A fetch from the page carries Origin: http://127.0.0.1:<port>."""
    client, token = api
    res = client.get("/api/defaults",
                     headers=_ok(token, Origin=f"http://127.0.0.1:{PORT}"))
    assert res.status_code == 200


def test_localhost_host_header_is_allowed(api):
    client, token = api
    res = client.get("/api/defaults",
                     headers={"X-Omnilingual-Token": token,
                              "Host": f"localhost:{PORT}"})
    assert res.status_code == 200


@pytest.mark.parametrize(("method", "path"), RESERVED)
def test_every_reserved_route_rejects_a_request_without_the_token(api, method, path):
    client, _ = api
    res = client.request(method, path, headers={"Host": HOST})
    assert res.status_code == 403, f"{method} {path} answered {res.status_code}"


def test_the_route_table_is_exactly_the_specified_one(api):
    """A new /api route must be listed in RESERVED to be covered by the guard test."""
    declared = {(method, route.path)
                for route in api[0].app.routes
                if route.path.startswith("/api/")
                for method in getattr(route, "methods", set())}
    assert declared == set(RESERVED) | {("GET", "/api/health")}


def test_no_cors_middleware_is_installed(api):
    """CORS would hand a foreign page the read it is refused here."""
    client, token = api
    names = [m.cls.__name__ for m in client.app.user_middleware]
    assert not any("CORS" in name for name in names), names
    allowed = client.get("/api/defaults", headers=_ok(token))
    assert "access-control-allow-origin" not in allowed.headers
    preflight = client.options("/api/defaults", headers={
        "Host": HOST, "Origin": "https://evil.example",
        "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in preflight.headers


def test_the_page_and_the_token_script_are_reachable_without_a_token(api):
    """Both are same-origin, read-only, and needed before the page has a token."""
    client, token = api
    page = client.get("/", headers={"Host": HOST})
    assert page.status_code == 200
    script = client.get("/token.js", headers={"Host": HOST})
    assert script.status_code == 200
    assert token in script.text


def test_a_foreign_host_cannot_read_the_page_or_the_token_script(api):
    client, token = api
    for path in ("/", "/token.js"):
        res = client.get(path, headers={"Host": "evil.example"})
        assert res.status_code == 403, path
        assert token not in res.text


def test_mint_token_is_unguessable_and_fresh():
    first, second = mint_token(), mint_token()
    assert first != second
    assert len(first) >= 32
    assert "/" not in first and "+" not in first


# --- defaults and settings -------------------------------------------------


def test_defaults_include_providers_and_models(api):
    client, token = api
    body = client.get("/api/defaults", headers=_ok(token)).json()
    assert "sarvam" in body["stt_providers"]
    assert "mayura" in body["mt_providers"]
    assert body["default_stt_models"]["sarvam"] == "saaras:v4"
    assert body["device"] == "Omnilingual"
    assert body["max_chunk_s"] == 28.0


def test_defaults_carry_every_live_option_and_the_output_directory(api):
    client, token = api
    body = client.get("/api/defaults", headers=_ok(token)).json()
    from omnilingual.ui import settings

    for field in ("device", "mic_only", "target_s", "max_chunk_s", "min_chunk_s",
                  "noise_db", "stt_workers", "max_cost", "work_dir",
                  "english_only", "out", "source", "langs", "num_speakers"):
        assert field in body, field
        assert field in settings.DEFAULTS, field
    assert body["last_output_dir"] == str(server.REPO_ROOT)


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


def test_settings_reject_a_body_that_is_not_an_object(api):
    client, token = api
    res = client.put("/api/settings", headers=_ok(token), json=["out"])
    assert res.status_code == 400
    assert "object" in res.json()["error"]


def test_settings_reject_a_malformed_body(api):
    client, token = api
    res = client.put("/api/settings",
                     headers={**_ok(token), "Content-Type": "application/json"},
                     content=b"{not json")
    assert res.status_code == 400
    assert "JSON" in res.json()["error"]


def test_a_corrupt_settings_store_does_not_stop_the_app(api):
    from omnilingual.ui import settings

    settings.settings_path().parent.mkdir(parents=True, exist_ok=True)
    settings.settings_path().write_text("this is not = valid = toml [[[",
                                        encoding="utf-8")
    client, token = api
    res = client.get("/api/defaults", headers=_ok(token))
    assert res.status_code == 200
    assert res.json()["out"] == settings.DEFAULTS["out"]


# --- API keys --------------------------------------------------------------


def test_keys_endpoint_returns_booleans_only(api, env):
    env.write_text(f"GROQ_API_KEY={FAKE_SECRET}\n", encoding="utf-8")
    client, token = api
    res = client.get("/api/keys", headers=_ok(token))
    assert res.json() == {"SARVAM_API_KEY": False, "GROQ_API_KEY": True,
                          "GEMINI_API_KEY": False}
    assert FAKE_SECRET not in res.text


def test_put_keys_writes_and_clear_removes(api, env):
    client, token = api
    assert client.put("/api/keys", headers=_ok(token),
                      json={"GROQ_API_KEY": "written-by-test"}).status_code == 200
    assert "written-by-test" in env.read_text(encoding="utf-8")
    assert client.put("/api/keys", headers=_ok(token),
                      json={"GROQ_API_KEY": None}).status_code == 200
    assert "GROQ_API_KEY" not in env.read_text(encoding="utf-8")


def test_put_keys_answers_with_the_new_presence_map(api, env):
    client, token = api
    body = client.put("/api/keys", headers=_ok(token),
                      json={"GROQ_API_KEY": FAKE_SECRET}).json()
    assert body == {"SARVAM_API_KEY": False, "GROQ_API_KEY": True,
                    "GEMINI_API_KEY": False}


def test_put_keys_rejects_an_unknown_name(api):
    client, token = api
    res = client.put("/api/keys", headers=_ok(token),
                     json={"AWS_SECRET_ACCESS_KEY": "x"})
    assert res.status_code == 400


@pytest.mark.parametrize("value", [f"{FAKE_SECRET}\nGEMINI_API_KEY=injected",
                                   f"{FAKE_SECRET}\rX", "  "])
def test_put_keys_rejects_a_rejected_value_without_echoing_it(api, env, value):
    """A value that cannot be stored must not come back in the 400 body."""
    client, token = api
    res = client.put("/api/keys", headers=_ok(token),
                     json={"GROQ_API_KEY": value})
    assert res.status_code in (200, 400)
    assert FAKE_SECRET not in res.text
    assert "injected" not in env.read_text(encoding="utf-8")


def test_put_keys_rejects_an_undecodable_env_without_echoing_the_value(api, env):
    env.write_bytes(b"SARVAM_API_KEY=\xff\xfe")
    client, token = api
    res = client.put("/api/keys", headers=_ok(token),
                     json={"GROQ_API_KEY": FAKE_SECRET})
    assert res.status_code == 400
    assert "UTF-8" in res.json()["error"]
    assert FAKE_SECRET not in res.text
    assert env.read_bytes() == b"SARVAM_API_KEY=\xff\xfe"


def test_put_keys_rejects_a_malformed_body_without_echoing_it(api):
    client, token = api
    res = client.put("/api/keys",
                     headers={**_ok(token), "Content-Type": "application/json"},
                     content=f'{{"GROQ_API_KEY": "{FAKE_SECRET}"'.encode())
    assert res.status_code == 400
    assert FAKE_SECRET not in res.text


@pytest.mark.parametrize("payload", [[FAKE_SECRET], FAKE_SECRET, 7, None])
def test_put_keys_rejects_a_body_that_is_not_an_object(api, payload):
    client, token = api
    res = client.put("/api/keys", headers=_ok(token), json=payload)
    assert res.status_code == 400
    assert FAKE_SECRET not in res.text


def test_a_rejected_key_never_reaches_a_traceback(env, monkeypatch):
    """An unexpected failure inside a writer must still answer 500 without the
    value. raise_server_exceptions off, because otherwise TestClient re-raises the
    exception into the test instead of producing the response a user would see."""
    from omnilingual.ui import secrets

    def explode(name, value):
        raise RuntimeError(f"boom {name} {value}")

    monkeypatch.setattr(secrets, "set_key", explode)
    token = mint_token()
    app = create_app(token=token, port=PORT)
    with TestClient(app, raise_server_exceptions=False) as client:
        res = client.put("/api/keys", headers=_ok(token),
                         json={"GROQ_API_KEY": FAKE_SECRET})
    assert res.status_code == 500
    assert FAKE_SECRET not in res.text


# --- audio readiness -------------------------------------------------------


def test_audio_endpoint_returns_the_readiness_shape(api, monkeypatch):
    from omnilingual.ui import audio

    monkeypatch.setattr(audio, "probe",
                        lambda *a, **k: _readiness(output="Speakers"))
    client, token = api
    body = client.get("/api/audio", headers=_ok(token)).json()
    assert body["ok"] is True
    assert body["device"] == "Omnilingual"
    assert body["mic_authorized"] is True


def test_audio_endpoint_says_the_microphone_was_actually_checked(api, monkeypatch):
    """mic_authorized is True for a skipped probe too, so the page needs this."""
    from omnilingual.ui import audio

    seen = {}

    def probe(*args, mic=True, **kwargs):
        seen["mic"] = mic
        return _readiness()

    monkeypatch.setattr(audio, "probe", probe)
    client, token = api
    body = client.get("/api/audio", headers=_ok(token)).json()
    assert seen["mic"] is True
    assert body["mic_checked"] is True


def test_audio_endpoint_warns_about_the_wrong_output_device(api, monkeypatch):
    from omnilingual.ui import audio

    monkeypatch.setattr(audio, "probe",
                        lambda *a, **k: _readiness(output="MacBook Pro Speakers"))
    client, token = api
    detail = " ".join(client.get("/api/audio", headers=_ok(token)).json()["detail"])
    assert "Multi-Output Device" in detail


def test_audio_endpoint_does_not_switch_the_output_device(api, monkeypatch):
    """The read path may report the output device; it must never change it."""
    from omnilingual.ui import audio

    def forbidden():
        raise AssertionError("the server switched the audio output device")

    monkeypatch.setattr(audio, "restart_daemon", forbidden)
    monkeypatch.setattr(audio, "setup", forbidden)
    monkeypatch.setattr(audio, "probe", lambda *a, **k: _readiness())
    client, token = api
    assert client.get("/api/audio", headers=_ok(token)).status_code == 200


def test_audio_setup_streams_progress_and_reports_the_mic_as_not_checked(api,
                                                                        monkeypatch):
    """Progress goes on the log channel; readiness is the page's own re-poll."""
    from omnilingual.ui import audio

    probes = []

    def probe(*args, mic=True, **kwargs):
        probes.append(mic)
        return _readiness()

    monkeypatch.setattr(audio, "probe", probe)
    monkeypatch.setattr(audio, "setup", lambda *a, **k: iter(["building", "ready"]))
    client, token = api
    res = client.post("/api/audio/setup", headers=_ok(token))
    assert res.status_code == 200
    assert _wait_for_logs(client, 2) == ["building", "ready"]
    # The route must not report a microphone nobody asked about; the page re-polls
    # /api/audio, which probes for real and says so with mic_checked.
    assert client.get("/api/audio", headers=_ok(token)).json()["mic_checked"] is True


def test_audio_setup_failure_is_reported(api, monkeypatch):
    """A detached operation cannot answer 500, so its failure is a log line."""
    from omnilingual.ui import audio

    def broken(*a, **k):
        raise RuntimeError("BlackHole never appeared")
        yield  # pragma: no cover - a generator, never reached

    monkeypatch.setattr(audio, "setup", broken)
    client, token = api
    res = client.post("/api/audio/setup", headers=_ok(token))
    assert res.status_code == 200
    # Named like SessionRunner._fail's, so the page can tell a refusal from a crash.
    assert _wait_for_logs(client, 1) == ["RuntimeError: BlackHole never appeared"]
    errors = [m for m in list(_state(client).queue._queue) if m["type"] == "log"]
    assert [m["level"] for m in errors] == ["error"]


def test_audio_restart_daemon_reports_a_refusal(api, monkeypatch):
    from omnilingual.ui import audio

    def refused():
        raise subprocess.CalledProcessError(1, "osascript")

    monkeypatch.setattr(audio, "restart_daemon", refused)
    client, token = api
    res = client.post("/api/audio/restart-daemon", headers=_ok(token))
    assert res.status_code == 200
    assert _wait_for_logs(client, 1)


# --- long operations must not hold the event loop ----------------------------
#
# §7 sends setup's and apply's progress on the WebSocket log channel, and this
# server has exactly one event loop. A handler that blocks on the work queues its
# own progress lines somewhere nothing can read until it returns, so the streaming
# is only a queue write, and no other request is served meanwhile. audio.setup
# waits up to 60s for BlackHole and restart-daemon waits on a GUI auth dialog for
# up to 120s, so this is minutes of an unresponsive app.


class _SlowProc:
    """A Popen stand-in that emits one line and then blocks, like a real install."""

    def __init__(self, release):
        self._release = release
        self._emitted = False
        self.stdout = self
        self.code = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self._emitted:
            self._emitted = True
            return "==> installing\n"
        self._release.wait(8)
        raise StopIteration

    def close(self):
        pass

    def wait(self, timeout=None):
        return self.code


def _block(route, monkeypatch, release):
    """Make the operation behind `route` block on `release` after it starts."""
    from omnilingual.ui import audio

    if route == "/api/audio/setup":
        def setup(*a, **k):
            yield "building"
            release.wait(8)
        monkeypatch.setattr(audio, "setup", setup)
    elif route == "/api/audio/restart-daemon":
        def restart():
            release.wait(8)
        monkeypatch.setattr(audio, "restart_daemon", restart)
    else:
        monkeypatch.setattr(server.subprocess, "Popen",
                            lambda argv, **kw: _SlowProc(release))


@pytest.mark.parametrize("route", ["/api/audio/setup", "/api/audio/restart-daemon",
                                   "/api/setup/apply"])
def test_a_long_operation_never_holds_the_event_loop(api, monkeypatch, route):
    release = threading.Event()
    _block(route, monkeypatch, release)
    client, token = api
    started = time.monotonic()
    res = client.post(route, headers=_ok(token))
    elapsed = time.monotonic() - started
    try:
        # Still serving while the operation is in flight: this request would queue
        # behind it if the handler had not returned yet.
        assert client.get("/api/health", headers={"Host": HOST}).status_code == 200
    finally:
        release.set()
    assert res.status_code == 200
    assert elapsed < 2, (
        f"{route} blocked its own event loop for {elapsed:.1f}s, so the progress "
        "it queues cannot reach a page until it finishes")


def test_a_long_operation_runs_on_a_daemon_thread(api, monkeypatch):
    """A quit during an install must not wait for brew."""
    from omnilingual.ui import audio

    seen = {}

    def setup(*a, **k):
        thread = threading.current_thread()
        seen["thread"] = thread
        yield "done"

    monkeypatch.setattr(audio, "setup", setup)
    client, token = api
    client.post("/api/audio/setup", headers=_ok(token))
    assert _wait_for_logs(client, 1) == ["done"]
    thread = seen.get("thread")
    assert thread is not None
    assert thread.daemon is True
    assert thread is not threading.main_thread()


# --- starting a run --------------------------------------------------------


def test_start_rejects_an_unknown_stt_provider(api):
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(stt="not-a-provider"))
    assert res.status_code == 400
    assert "not-a-provider" in res.json()["error"]


def test_start_rejects_a_missing_api_key(api):
    """The credential gate in build_run, before any thread could hang on it."""
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body())
    assert res.status_code == 400
    assert "SARVAM_API_KEY" in res.json()["error"]


def test_start_rejects_bad_chunk_bounds(api, env):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(min_chunk_s=5.0, max_chunk_s=40.0))
    assert res.status_code == 400
    assert "--max-chunk-s must be < 30" in res.json()["error"]


def test_start_rejects_a_target_outside_the_bounds(api, env):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(target_s=99.0))
    assert res.status_code == 400
    assert "--target-s" in res.json()["error"]


def test_start_rejects_a_field_that_is_not_a_number(api, env):
    """The shared numeric wrapper owns this message, so the server has no _as_float."""
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(min_chunk_s="soon"))
    assert res.status_code == 400
    assert "--min-chunk-s" in res.json()["error"]


def test_start_rejects_a_cleared_numeric_field_by_using_the_default(fake_runs, env):
    """A field the user emptied is absent, not an error."""
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(target_s=None, max_chunk_s=""))
    assert res.status_code == 200
    opts = _runner(client).calls[0][1]["opts"]
    assert opts.target_s == 8.0
    assert opts.max_chunk_s == 28.0


def test_start_rejects_an_unknown_panel_field(api, env):
    """One shared allowlist: a field the panel does not own is a bad request."""
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(spoken_language="Hindi"))
    assert res.status_code == 400
    assert "spoken_language" in res.json()["error"]


def test_start_rejects_a_mode_that_is_not_a_mode(api):
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(mode="watching"))
    assert res.status_code == 400
    assert "mode" in res.json()["error"]


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


def test_start_rejects_a_malformed_body(api):
    client, token = api
    res = client.post("/api/session/start",
                      headers={**_ok(token), "Content-Type": "application/json"},
                      content=b"[oops")
    assert res.status_code == 400
    assert "JSON" in res.json()["error"]


def test_start_reports_a_missing_ffmpeg_before_it_starts_a_run(api, env, monkeypatch):
    from omnilingual.audio.normalize import FfmpegMissingError

    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    monkeypatch.setattr(server, "SessionRunner", FakeRunner)

    def missing():
        raise FfmpegMissingError("ffmpeg not found: brew install ffmpeg")

    monkeypatch.setattr(server, "ensure_ffmpeg", missing)
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token), json=_start_body())
    assert res.status_code == 400
    assert "ffmpeg" in res.json()["error"]
    assert _state(client).runs == {}


def test_start_refuses_a_live_run_when_the_device_is_absent(fake_runs, env,
                                                             monkeypatch):
    """§9: a live start that cannot capture is a 400 naming the remedy.

    The alternative is answering 200 and letting the run thread fail seconds
    later with an opaque ffmpeg error, by which point the panel looks broken.
    """
    from omnilingual.ui import audio

    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    monkeypatch.setattr(audio, "probe", lambda *a, **k: _readiness(device=False))
    client, token = fake_runs
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body())
    assert res.status_code == 400
    error = res.json()["error"]
    assert "Omnilingual" in error
    assert "Set up audio" in error
    assert _state(client).runs == {}


def test_start_refuses_a_live_run_when_blackhole_is_absent(fake_runs, env,
                                                           monkeypatch):
    from omnilingual.ui import audio

    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    monkeypatch.setattr(audio, "probe",
                        lambda *a, **k: _readiness(blackhole=False))
    client, token = fake_runs
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body())
    assert res.status_code == 400
    assert "blackhole-2ch" in res.json()["error"].lower()


def test_start_refuses_a_live_run_when_the_microphone_is_denied(fake_runs, env,
                                                                monkeypatch):
    """The one-second capture is what answers this, so the probe must ask for it."""
    from omnilingual.ui import audio

    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    asked: list[bool] = []

    def probe(*args, **kwargs):
        asked.append(kwargs.get("mic"))
        return _readiness(mic_authorized=False)

    monkeypatch.setattr(audio, "probe", probe)
    client, token = fake_runs
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body())
    assert res.status_code == 400
    assert "Microphone" in res.json()["error"]
    assert asked == [True], "a mic=False probe would report a denial it never saw"
    assert _state(client).runs == {}


def test_the_live_gate_does_not_blame_the_microphone_it_never_asked(fake_runs, env,
                                                                    monkeypatch):
    """probe skips the microphone when the device is missing, so it reports False.

    Taking that False as a denial would send the user to System Settings for a
    permission that was never the problem.
    """
    from omnilingual.ui import audio

    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    monkeypatch.setattr(
        audio, "probe",
        lambda *a, **k: _readiness(device=False, mic_authorized=False))
    client, token = fake_runs
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body())
    assert res.status_code == 400
    assert "Microphone" not in res.json()["error"]


def test_the_live_gate_never_switches_the_output_device(fake_runs, env,
                                                        monkeypatch):
    """A wrong output is reported; it is never routed, because that breaks the
    volume keys."""
    from omnilingual.ui import audio

    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    monkeypatch.setattr(audio, "probe",
                        lambda *a, **k: _readiness(output="Speakers"))
    switched: list[str] = []
    monkeypatch.setattr(audio, "_current_output", lambda: switched.append("read")
                        or "Speakers")
    client, token = fake_runs
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body())
    # The wrong output device is a warning, not a refusal: capture still opens.
    assert res.status_code == 200
    assert switched == []


def test_a_panel_rejection_costs_no_hardware_probe(api, env, monkeypatch):
    """Validation runs first: a bad panel is answered from the payload alone."""
    from omnilingual.ui import audio

    def forbidden(*a, **k):
        raise AssertionError("the audio probe must not run for a rejected panel")

    monkeypatch.setattr(audio, "probe", forbidden)
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(stt="not-a-provider"))
    assert res.status_code == 400
    assert "not-a-provider" in res.json()["error"]


def test_a_recording_start_never_probes_the_microphone(fake_runs, env,
                                                       monkeypatch):
    """A recording reads a file. There is nothing to capture and nothing to ask."""
    from omnilingual.ui import audio

    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    monkeypatch.setattr(
        audio, "probe",
        lambda *a, **k: pytest.fail("a recording must not probe audio"))
    client, token = fake_runs
    source = pathlib.Path(env.parent / "meeting.m4a")
    source.write_bytes(b"not really audio")
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(mode="recording", source=str(source)))
    assert res.status_code == 200


def test_start_builds_a_live_run_through_build_run(fake_runs, env, tmp_path):
    """The one path that may construct a run: build_run, then live_options."""
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    out = tmp_path / "notes.md"
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(out=str(out), target_s=9.0,
                                       max_chunk_s=27.0, min_chunk_s=4.0,
                                       noise_db=-30.0, stt_workers=3,
                                       max_cost=12.5, mic_only=True,
                                       english_only=True))
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "live"
    assert body["run_id"]
    runner = _runner(client)
    kind, kwargs = runner.calls[0]
    assert kind == "live"
    assert isinstance(kwargs["stt"], SarvamSTT)
    assert isinstance(kwargs["translator"], MayuraTranslator)
    assert kwargs["diarizer"] is None
    assert kwargs["settings"].stt_provider == "sarvam"
    opts = kwargs["opts"]
    assert opts.out == out
    assert opts.device == "Omnilingual"
    assert (opts.target_s, opts.max_chunk_s, opts.min_chunk_s) == (9.0, 27.0, 4.0)
    assert (opts.noise_db, opts.stt_workers, opts.max_cost) == (-30.0, 3, 12.5)
    assert opts.mic_only is True
    assert opts.english_only is True
    assert opts.work_root is None


def test_start_routes_a_recording_to_start_recording(fake_runs, env, tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"not really audio")
    out = tmp_path / "out" / "meeting.md"
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(mode="recording", source=str(source),
                                       out=str(out), english_only=True))
    assert res.status_code == 200
    assert res.json()["mode"] == "recording"
    kind, kwargs = _runner(client).calls[0]
    assert kind == "recording"
    assert kwargs["source"] == source
    assert kwargs["out"] == out
    assert kwargs["work_root"] == out.parent / ".omnilingual"
    assert kwargs["english_only"] is True


def test_start_defaults_a_recordings_output_to_the_recordings_own_name(fake_runs, env,
                                                                        tmp_path):
    """The CLI's rule: --out defaults to <recording>.md, not to a shared name."""
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    source = tmp_path / "standup.m4a"
    source.write_bytes(b"x")
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(mode="recording", source=str(source),
                                       out=""))
    assert res.status_code == 200
    assert _runner(client).calls[0][1]["out"] == source.with_suffix(".md")


def test_start_uses_the_stored_work_dir_for_a_recording(fake_runs, env, tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"x")
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(mode="recording", source=str(source),
                                       out=str(tmp_path / "o.md"),
                                       work_dir=str(tmp_path / "wd")))
    assert res.status_code == 200
    assert _runner(client).calls[0][1]["work_root"] == tmp_path / "wd"


def test_start_passes_a_work_dir_through_to_the_live_options(fake_runs, env, tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(out=str(tmp_path / "o.md"),
                                       work_dir=str(tmp_path / "wd")))
    assert res.status_code == 200
    assert _runner(client).calls[0][1]["opts"].work_root == tmp_path / "wd"


def test_start_resolves_a_relative_output_against_the_repo(fake_runs, env):
    """A Finder launch has no useful CWD, so a bare filename lands in the repo."""
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(out="standup.md"))
    assert res.status_code == 200
    assert _runner(client).calls[0][1]["opts"].out == server.REPO_ROOT / "standup.md"


def test_start_does_not_rename_an_existing_output(fake_runs, env, tmp_path):
    """SessionRunner.start_live owns the collision rename; a second one here
    would report a filename the run never writes."""
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    out = tmp_path / "taken.md"
    out.write_text("earlier transcript", encoding="utf-8")
    client, token = fake_runs
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(out=str(out)))
    assert res.status_code == 200
    assert _runner(client).calls[0][1]["opts"].out == out


def test_start_never_echoes_a_key(fake_runs, env):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    res = client.post("/api/session/start", headers=_ok(token), json=_start_body())
    assert res.status_code == 200
    assert FAKE_KEY not in res.text


def test_start_answers_409_while_a_run_is_active(fake_runs, env, tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    body = _start_body(out=str(tmp_path / "a.md"))
    assert client.post("/api/session/start", headers=_ok(token),
                       json=body).status_code == 200
    again = client.post("/api/session/start", headers=_ok(token),
                        json=_start_body(out=str(tmp_path / "b.md")))
    assert again.status_code == 409
    assert len(_state(client).runs) == 1


def test_a_finished_run_frees_the_window_again(fake_runs, env, tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    first = client.post("/api/session/start", headers=_ok(token),
                        json=_start_body(out=str(tmp_path / "a.md")))
    runner = _runner(client)
    runner.status = "done"
    second = client.post("/api/session/start", headers=_ok(token),
                         json=_start_body(out=str(tmp_path / "b.md")))
    assert second.status_code == 200
    assert len(_state(client).runs) == 2
    assert first.json()["run_id"] != second.json()["run_id"]


def test_a_halted_run_frees_the_window_but_stays_recoverable(fake_runs, env,
                                                             tmp_path):
    """halted is terminal: SessionRunner only reports it once the run has returned.
    What it adds to 'done' is that the session dir holds sealed chunks worth
    replaying, which is what recoverable says."""
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    started = client.post("/api/session/start", headers=_ok(token),
                          json=_start_body(out=str(tmp_path / "a.md"))).json()
    runner = _runner(client)
    runner.status = "halted"
    runner._session_dir = tmp_path / "live-x"
    runner._recoverable = True
    again = client.post("/api/session/start", headers=_ok(token),
                        json=_start_body(out=str(tmp_path / "b.md")))
    assert again.status_code == 200
    runs = client.get("/api/runs", headers=_ok(token)).json()["runs"]
    halted = [r for r in runs if r["run_id"] == started["run_id"]][0]
    assert halted["status"] == "halted"
    assert halted["recoverable"] is True


def test_run_ids_are_unique_per_run(fake_runs, env, tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    ids = []
    for name in "abc":
        res = client.post("/api/session/start", headers=_ok(token),
                          json=_start_body(out=str(tmp_path / f"{name}.md")))
        ids.append(res.json()["run_id"])
        for runner in _state(client).runs.values():
            runner.status = "done"
    assert len(set(ids)) == 3


# --- stopping a run --------------------------------------------------------


def test_stop_without_a_run_is_not_found(api):
    client, token = api
    res = client.post("/api/session/stop", headers=_ok(token))
    assert res.status_code == 404


def test_stop_after_the_run_finished_is_not_found(fake_runs, env, tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    client.post("/api/session/start", headers=_ok(token),
                json=_start_body(out=str(tmp_path / "a.md")))
    _runner(client).status = "done"
    assert client.post("/api/session/stop", headers=_ok(token)).status_code == 404


def test_stop_asks_a_live_run_to_stop(fake_runs, env, tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    started = client.post("/api/session/start", headers=_ok(token),
                          json=_start_body(out=str(tmp_path / "a.md"))).json()
    res = client.post("/api/session/stop", headers=_ok(token))
    assert res.status_code == 200
    assert res.json() == {"run_id": started["run_id"], "status": "stopping"}


def test_stop_does_not_invent_a_stopping_state_for_a_recording(fake_runs, env,
                                                               tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"x")
    client, token = fake_runs
    client.post("/api/session/start", headers=_ok(token),
                json=_start_body(mode="recording", source=str(source),
                                 out=str(tmp_path / "a.md")))
    body = client.post("/api/session/stop", headers=_ok(token)).json()
    assert body["status"] == "running"


# --- runs and recovery -----------------------------------------------------


def test_runs_endpoint_lists_nothing_before_a_run(api):
    client, token = api
    assert client.get("/api/runs", headers=_ok(token)).json()["runs"] == []


def test_runs_endpoint_reports_the_run(fake_runs, env, tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    started = client.post("/api/session/start", headers=_ok(token),
                          json=_start_body(out=str(tmp_path / "a.md"))).json()
    runs = client.get("/api/runs", headers=_ok(token)).json()["runs"]
    assert [r["run_id"] for r in runs] == [started["run_id"]]
    assert runs[0]["status"] == "running"
    assert runs[0]["out"] == str(tmp_path / "a.md")
    assert runs[0]["recoverable"] is False


def test_recover_rejects_a_directory_that_is_not_a_session_dir(api, tmp_path):
    client, token = api
    res = client.post("/api/session/recover", headers=_ok(token),
                      json={"session_dir": str(tmp_path / "nope")})
    assert res.status_code == 400
    assert "not a session dir" in res.json()["error"]


def test_recover_requires_a_session_dir(api):
    client, token = api
    res = client.post("/api/session/recover", headers=_ok(token), json={})
    assert res.status_code == 400


def test_recover_routes_to_run_from_chunks(fake_runs, env, tmp_path):
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    session_dir = tmp_path / "live-20260101T000000000000Z"
    session_dir.mkdir()
    res = client.post("/api/session/recover", headers=_ok(token),
                      json={"session_dir": str(session_dir),
                            "out": str(tmp_path / "again.md")})
    assert res.status_code == 200
    assert res.json()["mode"] == "recover"
    kind, kwargs = _runner(client).calls[0]
    assert kind == "recover"
    assert kwargs["session_dir"] == session_dir
    assert kwargs["out"] == tmp_path / "again.md"
    assert isinstance(kwargs["stt"], SarvamSTT)
    assert isinstance(kwargs["translator"], MayuraTranslator)


def test_recover_needs_no_source_file(fake_runs, env, tmp_path):
    """A recovery replays saved chunks, so it must not be refused for a missing
    source the way a recording is."""
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    session_dir = tmp_path / "live-x"
    session_dir.mkdir()
    client, token = fake_runs
    res = client.post("/api/session/recover", headers=_ok(token),
                      json={"session_dir": str(session_dir),
                            "out": str(tmp_path / "again.md")})
    assert res.status_code == 200


def test_recover_answers_409_while_a_run_is_active(fake_runs, env, tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    session_dir = tmp_path / "live-x"
    session_dir.mkdir()
    client, token = fake_runs
    client.post("/api/session/start", headers=_ok(token),
                json=_start_body(out=str(tmp_path / "a.md")))
    res = client.post("/api/session/recover", headers=_ok(token),
                      json={"session_dir": str(session_dir)})
    assert res.status_code == 409


def test_recover_rejects_a_missing_key(api, tmp_path):
    session_dir = tmp_path / "live-x"
    session_dir.mkdir()
    client, token = api
    res = client.post("/api/session/recover", headers=_ok(token),
                      json={"session_dir": str(session_dir)})
    assert res.status_code == 400
    assert "SARVAM_API_KEY" in res.json()["error"]


# --- the setup script ------------------------------------------------------


class _FakeProc:
    """A Popen stand-in with a real file object, because the server closes it."""

    def __init__(self, lines, code=0):
        self.stdout = io.StringIO("".join(lines))
        self.code = code

    def wait(self, timeout=None):
        return self.code


def test_setup_preview_returns_the_scripts_own_text(api, monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="plan\n",
                                           stderr="==> step\n")

    monkeypatch.setattr(server.subprocess, "run", run)
    client, token = api
    res = client.get("/api/setup/preview", headers=_ok(token))
    assert res.status_code == 200
    assert "plan" in res.json()["output"]
    assert "==> step" in res.json()["output"]
    assert seen["argv"][0] == str(SCRIPT)
    assert "--dry-run" in seen["argv"]
    assert "--yes" in seen["argv"]


def test_setup_preview_reports_a_refusal(api, monkeypatch):
    def run(argv, **kwargs):
        raise subprocess.CalledProcessError(2, argv, stderr="boom")

    monkeypatch.setattr(server.subprocess, "run", run)
    client, token = api
    res = client.get("/api/setup/preview", headers=_ok(token))
    assert res.status_code == 500
    assert res.json()["error"]


def test_setup_apply_streams_on_the_log_channel(api, monkeypatch):
    """The script writes everything human-facing to stderr, so both are merged."""
    seen = {}

    def popen(argv, **kwargs):
        seen["argv"] = argv
        return _FakeProc(["==> installing\n", "==> done\n"])

    monkeypatch.setattr(server.subprocess, "Popen", popen)
    client, token = api
    res = client.post("/api/setup/apply", headers=_ok(token))
    assert res.status_code == 200
    assert "--dry-run" not in seen["argv"]
    assert "--yes" in seen["argv"]
    assert _wait_for_logs(client, 2) == ["==> installing", "==> done"]


def test_setup_apply_reports_a_failure(api, monkeypatch):
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda argv, **kw: _FakeProc(["==> failed\n"], code=1))
    client, token = api
    res = client.post("/api/setup/apply", headers=_ok(token))
    assert res.status_code == 200
    assert _wait_for_logs(client, 2) == [
        "==> failed", "RuntimeError: setup-mac.sh exited 1"]
    levels = [m["level"] for m in list(_state(client).queue._queue)
              if m["type"] == "log"]
    assert levels == ["info", "error"]


def test_setup_routes_take_no_path_from_the_request(api, monkeypatch):
    """Nothing a request sends may reach the argv of a root-privileged script."""
    seen = {}

    def popen(argv, **kwargs):
        seen["argv"] = argv
        return _FakeProc([])

    monkeypatch.setattr(server.subprocess, "Popen", popen)
    client, token = api
    client.post("/api/setup/apply", headers=_ok(token),
                json={"repo": "; touch /tmp/pwned"})
    assert _wait_until(lambda: "argv" in seen)
    assert seen["argv"] == [str(SCRIPT), "--yes"]


def test_the_setup_script_is_the_one_in_the_repo():
    assert SCRIPT == server.REPO_ROOT / "scripts" / "setup-mac.sh"
    assert SCRIPT.is_file()


# --- the event stream ------------------------------------------------------


def test_websocket_requires_the_token(api):
    client, _ = api
    with pytest.raises(Exception):
        with client.websocket_connect("/ws", headers={"Host": HOST}):
            pass


def test_websocket_rejects_a_foreign_host(api):
    client, token = api
    with pytest.raises(Exception):
        with client.websocket_connect("/ws", headers=_ok(token, Host="evil.example")):
            pass


def test_websocket_rejects_a_foreign_origin(api):
    """A cross-site WebSocket is not covered by the same-origin policy, so the
    handshake has to check Origin itself."""
    client, token = api
    with pytest.raises(Exception):
        with client.websocket_connect("/ws",
                                      headers=_ok(token, Origin="https://evil.example")):
            pass


def test_websocket_accepts_the_token_as_a_query_parameter(api):
    """The browser WebSocket API cannot set a header on the handshake."""
    client, token = api
    with client.websocket_connect(f"/ws?token={token}",
                                  headers={"Host": HOST}) as socket:
        assert socket.receive_json()["type"] == "state"


def test_websocket_sends_state_on_connect(api):
    client, token = api
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        message = socket.receive_json()
    assert message["type"] == "state"
    assert message["status"] in {"idle", "done", "failed"}
    assert message["cost_inr"] == 0.0
    assert message["cost_cap"] is None


def test_websocket_sends_the_running_state_on_connect(fake_runs, env, tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    client.post("/api/session/start", headers=_ok(token),
                json=_start_body(out=str(tmp_path / "a.md")))
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        message = socket.receive_json()
    assert message["status"] == "running"
    assert message["mode"] == "live"


def test_websocket_relays_queued_events(api):
    client, token = api
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        assert socket.receive_json()["type"] == "state"
        state = _state(client)
        for message in ({"type": "status", "message": "sealed #1"},
                        {"type": "segment", "seq": 0, "text": "hello"},
                        {"type": "log", "level": "warn", "message": "renamed"},
                        {"type": "end", "exit_code": 0, "recoverable": False}):
            state.loop.call_soon_threadsafe(state.queue.put_nowait, message)
        kinds = [socket.receive_json()["type"] for _ in range(4)]
    assert kinds == ["status", "segment", "log", "end"]


def test_a_websocket_disconnect_does_not_kill_the_run(fake_runs, env, tmp_path):
    """The run lives in the server; the page is only a reader of it."""
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    started = client.post("/api/session/start", headers=_ok(token),
                          json=_start_body(out=str(tmp_path / "a.md"))).json()
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        assert socket.receive_json()["status"] == "running"
    # The window closed mid-meeting; the run must still be there and still running.
    assert _state(client).runs[started["run_id"]].status == "running"
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        assert socket.receive_json()["status"] == "running"
    assert client.get("/api/runs", headers=_ok(token)).json()["runs"][0][
        "run_id"] == started["run_id"]


def test_events_emitted_with_no_page_attached_are_kept_for_the_next_one(fake_runs, env,
                                                                       tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    client.post("/api/session/start", headers=_ok(token),
                json=_start_body(out=str(tmp_path / "a.md")))
    state = _state(client)
    state.queue.put_nowait({"type": "log", "level": "info", "message": "missed"})
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        socket.receive_json()  # the state snapshot, first
        assert socket.receive_json()["message"] == "missed"


def test_websocket_stop_message_stops_the_run(fake_runs, env, tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    client.post("/api/session/start", headers=_ok(token),
                json=_start_body(out=str(tmp_path / "a.md")))
    runner = _runner(client)
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        socket.receive_json()
        socket.send_json({"type": "stop"})
    # Messages are handled in order, so the disconnect that ends the context can
    # only be observed after the stop was applied.
    assert runner.status == "stopping"


def test_websocket_stop_message_with_no_run_is_harmless(api):
    client, token = api
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        socket.receive_json()
        socket.send_json({"type": "stop"})
        socket.send_json({"type": "nonsense"})
    assert _state(client).runs == {}


def test_websocket_recover_message_restarts_a_session(fake_runs, env, tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    session_dir = tmp_path / "live-x"
    session_dir.mkdir()
    first = client.post("/api/session/start", headers=_ok(token),
                        json=_start_body(out=str(tmp_path / "a.md"))).json()
    runner = _runner(client)
    runner.status = "done"
    runner._session_dir = session_dir
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        socket.receive_json()
        socket.send_json({"type": "recover", "run_id": first["run_id"]})
    runs = _state(client).runs
    assert len(runs) == 2
    second = [r for r in runs.values() if r.run_id != first["run_id"]][0]
    assert second.calls[0][0] == "recover"
    assert second.calls[0][1]["session_dir"] == session_dir


def test_websocket_recover_message_for_an_unknown_run_is_ignored(api):
    client, token = api
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        socket.receive_json()
        socket.send_json({"type": "recover", "run_id": "run-does-not-exist"})
    assert _state(client).runs == {}


def test_websocket_survives_a_frame_that_is_not_json(api):
    """One malformed frame must not cost the page the rest of the transcript."""
    client, token = api
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        assert socket.receive_json()["type"] == "state"
        socket.send_text("not json at all")
        socket.send_json(["not", "an", "object"])
        state = _state(client)
        state.loop.call_soon_threadsafe(
            state.queue.put_nowait, {"type": "log", "level": "info",
                                     "message": "still here"})
        assert socket.receive_json()["message"] == "still here"


def test_the_queue_is_the_loop_the_socket_is_on(api):
    """Events are handed to the loop that owns the socket, from any thread."""
    client, token = api
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        socket.receive_json()
        loop = _state(client).loop
        assert isinstance(loop, asyncio.AbstractEventLoop)
        assert not loop.is_closed()


# --- packaging invariants this module owns ---------------------------------


def test_server_does_not_import_the_pipeline():
    """session.py is the only module allowed to know how a run executes."""
    source = pathlib.Path(server.__file__).read_text(encoding="utf-8")
    assert "omnilingual.pipeline" not in source


def test_server_imports_no_web_dependency_outside_itself():
    here = pathlib.Path(server.__file__).parent
    for path in sorted(here.glob("*.py")):
        if path.name == "server.py":
            continue
        source = path.read_text(encoding="utf-8")
        for lib in ("fastapi", "uvicorn", "pywebview"):
            assert f"import {lib}" not in source, path.name
            assert f"from {lib}" not in source, path.name