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
import re
import subprocess
import threading
import time

import pytest
pytest.importorskip("fastapi", reason="the ui extra is not installed")
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
    ("POST", "/api/setup/preview"),
    ("POST", "/api/setup/apply"),
    ("POST", "/api/relaunch"),
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
        self.stops = 0
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
        self.stops += 1
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


def _isolate_work_dir(client, token, tmp_path, *, name="work"):
    """Point the stored panel at a work directory this test owns.

    /api/runs reads the work directory on disk (§7: recent runs from the live-*
    session dirs), so a test that says what it expects to find has to say where it
    is looking. Without this, every runs test asserted on whatever the developer's
    own repo happened to hold — which is why the two below used to pass.
    """
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    client.put("/api/settings", headers=_ok(token), json={"work_dir": str(root)})
    return root


def _saved_session(root, stamp, *, chunk=True, marker=True):
    """A live-* session dir on disk, exactly as run_live leaves one."""
    directory = root / f"live-{stamp}"
    (directory / "live-chunks").mkdir(parents=True, exist_ok=True)
    if marker:
        (directory / "session.json").write_text('{"kind": "live"}', encoding="utf-8")
    if chunk:
        (directory / "live-chunks" / "0001.wav").write_bytes(b"RIFF")
    return directory


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


# A near-miss is the only thing that pins an exact comparison. "not-the-token" is
# not a near miss: it shares nothing with a real token, so it stays rejected under
# every way of comparing two strings loosely.
@pytest.mark.parametrize("truncation", [
    lambda t: t[:-1],          # one character short
    lambda t: t[:8],           # a prefix
    lambda t: t + "x",         # a token plus one character
    lambda t: t.upper(),       # case-folded
])
def test_a_token_that_is_nearly_right_is_forbidden(api, truncation):
    client, token = api
    res = client.get("/api/defaults", headers=_ok(truncation(token)))
    assert res.status_code == 403
    assert res.json()["error"] == "bad token"


def test_a_loopback_origin_on_another_port_is_rejected(api):
    """127.0.0.1 on a different port is a different origin, and some other local
    process could be serving it. A prefix match on the scheme would let this
    window's token travel there."""
    client, token = api
    res = client.get("/api/defaults",
                     headers=_ok(token, Origin="http://127.0.0.1:62345"))
    assert res.status_code == 403
    assert res.json()["error"] == "bad origin"


def test_a_host_header_on_another_port_is_rejected(api):
    client, token = api
    res = client.get("/api/defaults",
                     headers={"X-Omnilingual-Token": token,
                              "Host": "127.0.0.1:62345"})
    assert res.status_code == 403
    assert res.json()["error"] == "bad host"


def test_an_app_that_was_not_told_its_port_answers_a_normal_request(env):
    """create_app(port=None) is the launcher that does not know the port yet.

    Reading that as port 0 made every request demand "Host: 127.0.0.1:0" and the
    app refused everything it was ever sent — a silent brick, since it looked like
    a server that was up.
    """
    token = mint_token()
    client = TestClient(create_app(token=token, port=None))
    for host in (HOST, "localhost:5599", "127.0.0.1:62345"):
        res = client.get("/api/keys",
                         headers={"X-Omnilingual-Token": token, "Host": host})
        assert res.status_code == 200, host
        assert res.json() == {"SARVAM_API_KEY": False, "GROQ_API_KEY": False,
                              "GEMINI_API_KEY": False}


def test_an_app_with_no_port_is_still_loopback_only(env):
    token = mint_token()
    client = TestClient(create_app(token=token, port=None))
    for host in ("evil.example", "127.0.0.1", "attacker.example:5599"):
        res = client.get("/api/keys",
                         headers={"X-Omnilingual-Token": token, "Host": host})
        assert res.status_code == 403, host


def test_an_app_with_no_port_still_demands_its_token(env):
    token = mint_token()
    client = TestClient(create_app(token=token, port=None))
    assert client.get("/api/keys", headers={"Host": HOST}).status_code == 403
    assert client.get("/api/keys",
                      headers={"X-Omnilingual-Token": "no", "Host": HOST}
                      ).status_code == 403


def test_an_app_with_no_port_refuses_a_cross_port_loopback_origin(env):
    """The bug this pins: an app that was never told its port compared no port at
    all, so any loopback origin was accepted — a page on 127.0.0.1:62345 got 200
    and the token would have travelled with it.

    Without a port of its own the app still knows where the request arrived: the
    Host header. Same-origin means Origin names that authority, so a different port
    is refused. Host stays permissive because that is finding 6's fix and it cannot
    be narrowed until a launcher reports the bound port."""
    token = mint_token()
    client = TestClient(create_app(token=token, port=None))
    for origin in ("http://127.0.0.1:62345", "http://localhost:62345",
                   "http://evil.example:5599"):
        res = client.get("/api/defaults",
                         headers={"X-Omnilingual-Token": token, "Host": HOST,
                                  "Origin": origin})
        assert res.status_code == 403, origin
        assert res.json()["error"] == "bad origin", origin


def test_an_app_with_no_port_accepts_an_origin_that_matches_the_host(env):
    """The other half of the fix: same-origin still answers. The window fetches
    http://127.0.0.1:5599, so Origin and Host agree on the port and it is served —
    "port unknown" must not degrade into "refuse everything"."""
    token = mint_token()
    client = TestClient(create_app(token=token, port=None))
    for host, origin in ((HOST, "http://127.0.0.1:5599"),
                         ("localhost:5599", "http://localhost:5599"),
                         ("127.0.0.1:62345", "http://127.0.0.1:62345")):
        res = client.get("/api/keys",
                         headers={"X-Omnilingual-Token": token, "Host": host,
                                  "Origin": origin})
        assert res.status_code == 200, (host, origin)


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


def test_the_page_loads_its_own_assets_without_a_token(api):
    """The browser cannot attach a header to <link href> or <script src>.

    So an asset behind the token is a window that comes up blank, with a 403 the
    page has no way to read and therefore no way to retry past. These two files
    are the whole of what the tags in index.html load, and neither may carry the
    token — the exemption is a narrowing of who may read them, not of what they
    contain.
    """
    client, token = api
    for name in ("style.css", "app.js"):
        res = client.get(f"/static/{name}", headers={"Host": HOST})
        assert res.status_code == 200, (name, res.status_code, res.text[:120])
        assert token not in res.text, f"{name} leaks the token"
    # The exemption is the assets, not the token: /api/ is still guarded.
    assert client.get("/api/audio", headers={"Host": HOST}).status_code == 403


def test_the_static_exemption_does_not_relax_host_or_origin(api):
    """The exemption is the token check alone; the two guards run before it."""
    client, _ = api
    bad_host = client.get("/static/app.js", headers={"Host": "evil.example"})
    assert bad_host.status_code == 403, bad_host.text
    foreign = client.get("/static/app.js", headers={
        "Host": HOST, "Origin": "http://evil.example"})
    assert foreign.status_code == 403, foreign.text
    # A wrong token on an asset changes nothing, because none is asked for.
    wrong = client.get("/static/app.js", headers={
        "Host": HOST, "X-Omnilingual-Token": "not-the-token"})
    assert wrong.status_code == 200


def test_the_unauthenticated_rule_is_exact_names_plus_one_prefix(api):
    """Names would have to be re-edited per asset; a prefix is what holds.

    Also pins the trailing slash: "/static" and "/staticfoo" stay on the guarded
    side, so the rule cannot be widened by accident.
    """
    from omnilingual.ui.server import _unauthenticated

    for path in ("/", "/api/health", "/token.js",
                 "/static/style.css", "/static/app.js",
                 "/static/nested/thing.js"):
        assert _unauthenticated(path), path
    for path in ("/api/keys", "/api/audio", "/api/session/start",
                 "/api/settings", "/api/healthz", "/static", "/staticfoo/app.js",
                 "/token.js.map"):
        assert not _unauthenticated(path), path


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


def test_settings_reject_a_non_finite_number(api):
    """1e400 parses as inf, which the route's own serializer refuses — so the
    answer is 400, not a 500.

    Posted as raw bytes on purpose: httpx's json encoder refuses to serialize inf,
    so json= would test the client rather than the server. The literal is what a
    browser sends, since JSON.parse accepts 1e400 happily.
    """
    client, token = api
    res = client.put("/api/settings", headers=_ok(token),
                     content=b'{"max_cost": 1e400}')
    assert res.status_code == 400
    assert "max_cost" in res.json()["error"]


def test_a_rejected_setting_does_not_touch_the_store(api):
    """The blocker: save() used to write the file and *then* let the response
    serializer fail, so a 500 left `max_cost = inf` on disk. load() reads inf back
    happily, which made every later save fail too — only deleting ui.toml
    recovered. So the bytes must be identical before and after the refusal."""
    from omnilingual.ui import settings

    client, token = api
    client.put("/api/settings", headers=_ok(token), json={"out": "/tmp/kept.md"})
    before = settings.settings_path().read_bytes()

    res = client.put("/api/settings", headers=_ok(token),
                     content=b'{"max_cost": 1e400}')
    assert res.status_code == 400
    assert settings.settings_path().read_bytes() == before

    # And the store is still usable: the poison did not land.
    good = client.put("/api/settings", headers=_ok(token), json={"out": "probe.md"})
    assert good.status_code == 200
    assert "inf" not in settings.settings_path().read_text(encoding="utf-8")


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
    res = client.post(route, headers=_ok(token), json={})
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


def _hide_extra(monkeypatch, package):
    """Make importlib.util.find_spec report one optional dependency as absent.

    find_spec is how all four local providers check for their extra, so this walks
    the real missing-extra path rather than a stubbed one: whatever the provider
    does about the absence is what the route has to answer.
    """
    import importlib.util

    real = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        return None if name == package else real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


# §13: "Missing extra — detected at start; renders as 'Run Setup to add speaker
# diarization', a button, not a traceback." A 500 is the one answer the page cannot
# turn into a button, so the contract is the status code and not the wording.
@pytest.mark.parametrize(("package", "panel"), [
    ("mlx_whisper", {"stt": "mlx-whisper"}),
    ("ctranslate2", {"mt": "indictrans2"}),
    ("sherpa_onnx", {"diarize": True}),
])
def test_a_missing_extra_is_a_refusal_not_a_traceback(api, env, monkeypatch, package,
                                                      panel):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    _hide_extra(monkeypatch, package)
    monkeypatch.setattr(server, "SessionRunner", FakeRunner)
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token),
                      json=_start_body(**panel))
    assert res.status_code == 400, res.text
    assert "extra" in res.json()["error"]
    assert _state(client).runs == {}


def test_an_import_error_from_the_builder_is_also_a_refusal(api, env, monkeypatch):
    """Belt and braces for the clause above.

    Each of today's four providers wraps the absence as a ConfigError, so the
    previous test cannot tell "the route answers a missing extra" from "the route
    happens to answer these four ConfigErrors". A provider that let its own
    ImportError out would be a 500, so the route has to catch the exception itself.
    """
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    monkeypatch.setattr(server, "SessionRunner", FakeRunner)

    def unbuildable(body):
        raise ImportError("No module named 'ctranslate2'")

    monkeypatch.setattr(server, "build_run", unbuildable)
    client, token = api
    res = client.post("/api/session/start", headers=_ok(token), json=_start_body())
    assert res.status_code == 400, res.text
    assert "ctranslate2" in res.json()["error"]
    assert _state(client).runs == {}


def test_recovery_refuses_a_missing_extra_rather_than_failing(api, env, tmp_path,
                                                              monkeypatch):
    """The same button on the recovery path: a replay needs its providers too."""
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    root = _isolate_work_dir(api[0], api[1], tmp_path)
    saved = _saved_session(root, "20261005T000000000000Z")

    def unbuildable(body):
        raise ModuleNotFoundError("No module named 'ctranslate2'")

    monkeypatch.setattr(server, "build_run", unbuildable)
    client, token = api
    res = client.post("/api/session/recover", headers=_ok(token),
                      json={"session_dir": str(saved)})
    assert res.status_code == 400, res.text
    assert "ctranslate2" in res.json()["error"]
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


def test_runs_endpoint_lists_nothing_before_a_run(api, tmp_path):
    client, token = api
    _isolate_work_dir(client, token, tmp_path)
    assert client.get("/api/runs", headers=_ok(token)).json()["runs"] == []


def test_runs_endpoint_reports_the_run(fake_runs, env, tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    _isolate_work_dir(client, token, tmp_path)
    started = client.post("/api/session/start", headers=_ok(token),
                          json=_start_body(out=str(tmp_path / "a.md"))).json()
    runs = client.get("/api/runs", headers=_ok(token)).json()["runs"]
    assert [r["run_id"] for r in runs] == [started["run_id"]]
    assert runs[0]["status"] == "running"
    assert runs[0]["out"] == str(tmp_path / "a.md")
    assert runs[0]["recoverable"] is False


def test_runs_endpoint_surfaces_a_session_a_previous_window_left_behind(api, tmp_path):
    """§7 asks for recent runs from the work directory's live-* session dirs.

    Without them §8's "recovery is available even after a quit" is unreachable:
    POST /api/session/recover needs a caller-supplied session_dir, and a freshly
    launched window's registry is empty.
    """
    client, token = api
    root = _isolate_work_dir(client, token, tmp_path)
    saved = _saved_session(root, "20261005T054512123456Z")
    entry, = client.get("/api/runs", headers=_ok(token)).json()["runs"]
    assert entry == {"run_id": saved.name, "status": "done", "out": None,
                     "session_dir": str(saved), "recoverable": True}


def test_runs_endpoint_lists_the_newest_session_first(api, tmp_path):
    client, token = api
    root = _isolate_work_dir(client, token, tmp_path)
    for stamp in ("20260101T000000000000Z", "20260601T000000000000Z",
                  "20261005T000000000000Z"):
        _saved_session(root, stamp)
    runs = client.get("/api/runs", headers=_ok(token)).json()["runs"]
    assert [r["run_id"] for r in runs] == [
        "live-20261005T000000000000Z", "live-20260601T000000000000Z",
        "live-20260101T000000000000Z"]


def test_a_saved_session_with_no_sealed_chunk_is_not_offered_for_recovery(api, tmp_path):
    """run_from_chunks refuses an empty manifest, so a Recover button that always
    fails is worse than none. session.json alone does not prove a sealed chunk."""
    client, token = api
    root = _isolate_work_dir(client, token, tmp_path)
    _saved_session(root, "20261005T000000000000Z", chunk=False)
    entry, = client.get("/api/runs", headers=_ok(token)).json()["runs"]
    assert entry["recoverable"] is False


def test_a_directory_without_session_json_is_not_a_session(api, tmp_path):
    client, token = api
    root = _isolate_work_dir(client, token, tmp_path)
    (root / "live-20261005T000000000000Z").mkdir()
    assert client.get("/api/runs", headers=_ok(token)).json()["runs"] == []


def test_a_saved_session_is_not_listed_next_to_the_run_that_owns_it(fake_runs, env,
                                                                   tmp_path):
    """The registry knows the live status and output path; the work tree knows only
    that a directory exists. The run's own entry has to win, once."""
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    root = _isolate_work_dir(client, token, tmp_path)
    saved = _saved_session(root, "20261005T000000000000Z")
    started = client.post("/api/session/start", headers=_ok(token),
                          json=_start_body(out=str(tmp_path / "a.md"))).json()
    _runner(client)._session_dir = saved
    runs = client.get("/api/runs", headers=_ok(token)).json()["runs"]
    assert [r["run_id"] for r in runs] == [started["run_id"]]
    assert runs[0]["status"] == "running"
    assert runs[0]["out"] == str(tmp_path / "a.md")


def test_the_saved_session_listing_is_capped(api, tmp_path):
    """A work directory gains a directory per meeting and nothing ever prunes it."""
    client, token = api
    root = _isolate_work_dir(client, token, tmp_path)
    newest = server._PAST_LIMIT + 5
    for day in range(newest):
        _saved_session(root, f"202601{day + 1:02d}T000000000000Z")
    runs = client.get("/api/runs", headers=_ok(token)).json()["runs"]
    assert len(runs) == server._PAST_LIMIT
    assert runs[0]["run_id"] == f"live-202601{newest:02d}T000000000000Z"


def test_recovery_reaches_a_session_from_before_this_window(fake_runs, env, tmp_path):
    """The whole point of listing the saved sessions: the page posts the session_dir
    it was shown and run_from_chunks replays it, with no run id from this process."""
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    root = _isolate_work_dir(client, token, tmp_path)
    saved = _saved_session(root, "20261005T000000000000Z")
    listed = [r for r in client.get("/api/runs", headers=_ok(token)).json()["runs"]
              if r["run_id"] == saved.name]
    res = client.post("/api/session/recover", headers=_ok(token),
                      json={"session_dir": listed[0]["session_dir"]})
    assert res.status_code == 200
    assert res.json()["mode"] == "recover"
    assert _runner(client).calls[0][0] == "recover"
    assert _runner(client).calls[0][1]["session_dir"] == saved


def test_recover_message_reaches_a_session_this_window_did_not_start(fake_runs, env,
                                                                     tmp_path):
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    root = _isolate_work_dir(client, token, tmp_path)
    saved = _saved_session(root, "20261005T000000000000Z")
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        socket.receive_json()
        socket.send_json({"type": "recover", "run_id": saved.name})
    assert _runner(client).calls[0][0] == "recover"


def test_recover_message_for_a_saved_session_with_no_chunks_is_ignored(api, tmp_path):
    client, token = api
    root = _isolate_work_dir(client, token, tmp_path)
    empty = _saved_session(root, "20261005T000000000000Z", chunk=False)
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        socket.receive_json()
        socket.send_json({"type": "recover", "run_id": empty.name})
    assert _state(client).runs == {}


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


def test_setup_preview_returns_real_script_output(api, monkeypatch, tmp_path):
    """No stub: the real script's dry-run text must reach the response body.

    A missing stdout capture in _setup_preview lets the script inherit the
    server's own stdout, so the response is {"output": ""} while the plan
    text lands in the server log. Dry-run changes nothing on disk.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    client, token = api
    res = client.get("/api/setup/preview", headers=_ok(token))
    assert res.status_code == 200
    assert "extras enabled" in res.json()["output"]


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
    res = client.post("/api/setup/apply", headers=_ok(token), json={})
    assert res.status_code == 200
    assert "--dry-run" not in seen["argv"]
    assert "--yes" in seen["argv"]
    assert _wait_for_logs(client, 2) == ["==> installing", "==> done"]


def test_setup_apply_reports_a_failure(api, monkeypatch):
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda argv, **kw: _FakeProc(["==> failed\n"], code=1))
    client, token = api
    res = client.post("/api/setup/apply", headers=_ok(token), json={})
    assert res.status_code == 200
    assert _wait_for_logs(client, 2) == [
        "==> failed", "RuntimeError: setup-mac.sh exited 1"]
    levels = [m["level"] for m in list(_state(client).queue._queue)
              if m["type"] == "log"]
    assert levels == ["info", "error"]


def test_setup_routes_take_no_path_from_the_request(api, monkeypatch):
    """Nothing a request sends may reach the argv of a root-privileged script.

    Unknown fields are refused before the script starts, so the refusal
    itself is the guarantee: the argv is never built.
    """
    seen = {}

    def popen(argv, **kwargs):
        seen["argv"] = argv
        return _FakeProc([])

    monkeypatch.setattr(server.subprocess, "Popen", popen)
    client, token = api
    res = client.post("/api/setup/apply", headers=_ok(token),
                      json={"repo": "; touch /tmp/pwned"})
    assert res.status_code == 400
    assert "argv" not in seen


def test_the_setup_script_is_the_one_in_the_repo():
    assert SCRIPT == server.REPO_ROOT / "scripts" / "setup-mac.sh"
    assert SCRIPT.is_file()


def _script_shell():
    """The launcher as text, read once per call so an edit is seen immediately."""
    return SCRIPT.read_text(encoding="utf-8")


def test_the_launcher_installs_the_ui_extra_by_default():
    """The human ruling: ui is opted IN, so a fresh run installs the app.

    A saved setup-mac.conf still overrides this (load_config), which is why the
    handoff tells an existing user how to add ui rather than the script rewriting
    their file — see the handoff assertions below.
    """
    body = _script_shell()
    # The literal, not every EXTRAS= line: the script reassigns the variable later
    # through sanitize() and prompt_extras().
    default = re.search(r'^EXTRAS="([^"]*)"$', body, re.M)
    assert default, "the launcher must ship a literal EXTRAS default"
    extras = default.group(1).split(",")
    assert "ui" in extras
    # And the two it already carried, so this cannot pass by swapping one out.
    assert {"local-stt", "diarize"} <= set(extras)


def test_the_ui_extra_stays_deselectable():
    """The human ruling forbids touching ALL_EXTRAS, and this says why.

    ALL_EXTRAS is the union the convergence loop walks. A chosen extra is left out
    of the --no-extra list and so gets installed; an unchosen one is subtracted.
    Drop ui from ALL_EXTRAS and nothing can ever subtract it again — `uv sync
    --all-extras` installs the app and a later run cannot remove it, which is the
    permanent-install failure the ruling rules out. ui is already there, so the
    assertion holds it in place.
    """
    body = _script_shell()
    all_extras = re.search(r'^ALL_EXTRAS="([^"]*)"$', body, re.M)
    assert all_extras, "ALL_EXTRAS must be a literal"
    assert "ui" in all_extras.group(1).split()

    # And the loop must subtract rather than only add: the sync line has to carry
    # NO_EXTRA_ARGS, not be a bare `uv sync --all-extras`.
    sync_line = next(line for line in body.splitlines()
                     if "uv sync" in line and "NO_EXTRA_ARGS" in line)
    assert not sync_line.strip().endswith("uv sync --all-extras"), (
        "a bare --all-extras would install every extra permanently")
    build = body[body.index("NO_EXTRA_ARGS=()"):body.index(sync_line)]
    assert re.search(r'in_list "\$e" "\$EXTRAS" \|\| NO_EXTRA_ARGS\+=\(', build)


def test_the_launcher_tells_a_ui_user_to_run_the_app():
    body = _script_shell()
    handoff = body[body.index("--- handoff"):]
    assert "omnilingual-ui" in handoff


CAPABILITIES = {"extras": "diarize", "live_setup": False, "route_output": False,
                "prefetch": True, "run_tests": False}


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
    """`keys` is not an accepted field: the screen must never hold a secret.

    The refusal happens before the script starts, so no argv exists to inspect.
    """
    client, token = api
    seen: list[list[str]] = []
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda argv, **kw: (seen.append(list(argv)), _FakeProc(()))[1])

    res = client.post("/api/setup/apply", headers=_ok(token),
                      json={**CAPABILITIES, "keys": "SARVAM_API_KEY=sk-live-abc"})

    assert res.status_code == 400
    assert seen == []


def test_relaunch_reexecutes_a_fixed_argv(api, monkeypatch):
    """The relaunch must not take its command from the request.

    The exec is deferred past the response (the process image is replaced,
    so answering first is the only way the page learns the restart began),
    hence the poll rather than an immediate assertion. A daemon thread
    carries the delay: loop timers scheduled during a request never fire
    under TestClient, so call_later would be unobservable.
    """
    client, token = api
    calls: list[list[str]] = []
    monkeypatch.setattr(server, "_relaunch_argv", lambda: ["python", "-m", "omnilingual.ui"])
    monkeypatch.setattr(server.os, "execv", lambda path, argv: calls.append(list(argv)))

    response = client.post("/api/relaunch", headers=_ok(token))

    assert response.status_code == 200
    assert response.json() == {"restarting": True}
    assert _wait_until(lambda: calls == [["python", "-m", "omnilingual.ui"]])


def test_relaunch_is_token_guarded(api):
    client, _ = api
    assert client.post("/api/relaunch").status_code == 403


def test_the_launcher_explains_how_to_add_ui_to_a_saved_config():
    """load_config lets a saved EXTRAS= win over the default, so a user who ran the
    launcher before this change sees no ui at all. The note has to name both the
    file and the key, or it is not actionable."""
    body = _script_shell()
    handoff = body[body.index("--- handoff"):]
    assert "EXTRAS=" in handoff
    assert "setup-mac.conf" in handoff


def test_prefetch_label_knows_about_ui():
    """prefetch_label had no ui arm and was correct only because the prefetchable
    gate sits between the check and the call. The arm is the explicit form: ui
    ships wheels, not weights, so it has no label and must not claim a prefetch."""
    body = _script_shell()
    label_arm = body[body.index("prefetch_label() {"):body.index("prefetch_models()")]
    assert re.search(r"^\s+ui\)", label_arm, re.M), (
        "prefetch_label needs a ui arm so a dropped gate cannot print a lie")
    assert "return 1" in label_arm


def test_no_variable_expansion_runs_into_a_non_ascii_byte():
    """bash 3.2 takes the byte after `$name` as part of the name, so `"$label…"`
    looks up `label\\xef` and `set -u` aborts the script. It did exactly that at the
    first model prefetch, so on macOS's system bash the launcher died before its
    own handoff — including the ui line added in this wave. `${name}` is the fix."""
    import re as _re

    offenders = [
        line for line in _script_shell().splitlines()
        if _re.search(r"\$[A-Za-z_][A-Za-z0-9_]*[^\x00-\x7f]", line)
        # The in-script usage banner quotes these lines on purpose.
        and not line.lstrip().startswith(("#", "'"))
    ]
    assert offenders == [], (
        "brace these, or bash 3.2 reads the next byte as part of the name: "
        + "; ".join(offenders))


def test_a_preview_does_not_hold_the_event_loop(env, monkeypatch):
    """The preview runs a root-capable script synchronously and can take minutes.
    This server has one event loop, so blocking it stalls every other request,
    including the WebSocket a live run is streaming its rows over.

    Driven as raw ASGI on one asyncio loop rather than through TestClient, on
    purpose: TestClient gives each request its own portal and loop, so two
    TestClient calls are never actually concurrent and the test would pass even
    with the blocking call restored.
    """
    entered = threading.Event()
    release = threading.Event()
    blocked_at = []

    def run(argv, **kwargs):
        blocked_at.append(time.monotonic())
        entered.set()
        # Long enough that a blocked loop is unmistakable, short enough that a
        # failing run of this test is not a minute of wall clock.
        release.wait(timeout=5.0)
        return subprocess.CompletedProcess(argv, 0, stdout="plan\n", stderr="")

    monkeypatch.setattr(server.subprocess, "run", run)
    token = mint_token()
    app = create_app(token=token, port=PORT)

    async def call(path):
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                 "method": "GET", "scheme": "http", "path": path,
                 "raw_path": path.encode(), "query_string": b"",
                 "root_path": "", "headers": [
                     (b"host", HOST.encode()),
                     (b"x-omnilingual-token", token.encode())],
                 "client": ("127.0.0.1", 12345), "server": (HOST, PORT)}
        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        await asyncio.wait_for(app(scope, receive, send), timeout=15.0)
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        return status

    async def main():
        preview = asyncio.ensure_future(call("/api/setup/preview"))
        for _ in range(200):  # wait for the blocking call to actually be entered
            if entered.is_set():
                break
            await asyncio.sleep(0.005)
        assert entered.is_set(), "the preview never started"
        health = await call("/api/health")
        # Measured from inside the blocking call, not from here: if the loop is
        # blocked, *this* coroutine cannot run either, so timing from the line
        # below would start measuring only after the stall had already ended and
        # the test would pass on the broken code.
        served_in = time.monotonic() - blocked_at[0]
        release.set()
        preview_status = await preview
        return health, served_in, preview_status

    health, served_in, preview_status = asyncio.run(main())
    assert health == 200
    assert preview_status == 200
    assert served_in < 1.0, (
        f"/api/health queued behind the preview for {served_in:.3f}s")


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


def test_websocket_rejects_a_cross_port_loopback_origin(api):
    """The handshake is the one request a cross-site page can make with no preflight,
    so it repeats the origin check — and a loopback origin on another port is a
    different origin, served by whatever else is listening on this machine."""
    client, token = api
    with pytest.raises(Exception):
        with client.websocket_connect("/ws",
                                      headers=_ok(token, Origin="http://127.0.0.1:62345")):
            pass


def test_websocket_rejects_a_cross_port_loopback_origin_when_the_port_is_unknown(env):
    """The handshake repeats the origin check itself, so it repeats the fix: with no
    port of its own this app used to open the socket with any loopback origin, and
    the handshake is the one request a cross-site page can make with no preflight."""
    token = mint_token()
    client = TestClient(create_app(token=token, port=None))
    with pytest.raises(Exception):
        with client.websocket_connect(
                "/ws", headers={"X-Omnilingual-Token": token, "Host": HOST,
                                "Origin": "http://127.0.0.1:62345"}):
            pass


def test_websocket_with_no_port_opens_for_its_own_origin(env):
    """The matching half: a same-origin handshake is still accepted."""
    token = mint_token()
    client = TestClient(create_app(token=token, port=None))
    with client.websocket_connect(
            "/ws", headers={"X-Omnilingual-Token": token, "Host": HOST,
                            "Origin": f"http://127.0.0.1:{PORT}"}) as socket:
        assert socket.receive_json()["type"] == "state"


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


async def _disconnect(app, *, token):
    """Hand the app one websocket connect and then a disconnect, and await its return.

    TestClient cannot be used for this. Its session context manager exits through
    close(1000) -> portal.call(cs.cancel), and that call resolves at
    task_status.started(): the app coroutine is cancelled from outside and never
    runs its own teardown. A test written against TestClient therefore asserts about
    whatever Starlette happens to await, not about this app's disconnect path.
    Calling the ASGI app directly and awaiting it is the only way to see that the
    handler finished its own way out.
    """
    scope = {
        "type": "websocket", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "scheme": "ws", "server": ("127.0.0.1", PORT),
        "client": ("127.0.0.1", 53123), "root_path": "", "path": "/ws",
        "raw_path": b"/ws", "query_string": b"",
        "headers": [(b"host", HOST.encode()),
                    (b"x-omnilingual-token", token.encode()),
                    (b"origin", f"http://127.0.0.1:{PORT}".encode())],
    }
    inbound = [{"type": "websocket.connect"},
               {"type": "websocket.disconnect", "code": 1000, "reason": ""}]
    sent = []

    async def receive():
        if inbound:
            return inbound.pop(0)
        # Nothing left to hand over: hold the handler open rather than spinning, so
        # a handler that never notices the disconnect fails on the timeout instead
        # of hanging the suite.
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    async def send(message):
        sent.append(message)

    await asyncio.wait_for(app(scope, receive, send), 10)
    return sent


def test_the_disconnect_path_never_touches_a_runner(fake_runs, env, tmp_path):
    """Constraint 6: closing the window must not destroy a transcription in progress.

    Asserted against the app's own disconnect path, which is why this drives the
    ASGI app rather than a TestClient socket — see _disconnect.
    """
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    client.post("/api/session/start", headers=_ok(token),
                json=_start_body(out=str(tmp_path / "a.md")))
    runner = _runner(client)
    sent = asyncio.run(_disconnect(client.app, token=token))
    assert [m["type"] for m in sent] == ["websocket.accept", "websocket.send"]
    assert runner.stops == 0
    assert runner.status == "running"
    assert _state(client).runs[runner.run_id] is runner


def test_a_reconnecting_page_still_finds_the_run(fake_runs, env, tmp_path):
    """The run lives in the server; the page is only a reader of it."""
    env.write_text(f"SARVAM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    client, token = fake_runs
    _isolate_work_dir(client, token, tmp_path)
    started = client.post("/api/session/start", headers=_ok(token),
                          json=_start_body(out=str(tmp_path / "a.md"))).json()
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        assert socket.receive_json()["status"] == "running"
    runner = _runner(client, started["run_id"])
    assert runner.status == "running"
    with client.websocket_connect("/ws", headers=_ok(token)) as socket:
        assert socket.receive_json()["status"] == "running"
    runs = client.get("/api/runs", headers=_ok(token)).json()["runs"]
    assert started["run_id"] in [r["run_id"] for r in runs]


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
    """Each web library has one legal home: fastapi in server.py, uvicorn and
    pywebview in the launcher, and nowhere else.

    The allowance is per dependency, so a module that may import uvicorn still
    may not import fastapi, and no third module may claim either. It was written
    when the package held server.py alone; tests/test_ui_packaging.py states the
    same rule including __main__.py, which is where the launcher is supposed to
    import uvicorn from, and the two must not disagree.
    """
    allowed = {"server.py": {"fastapi"},
               "__main__.py": {"uvicorn", "pywebview"}}
    here = pathlib.Path(server.__file__).parent
    for path in sorted(here.glob("*.py")):
        permitted = allowed.get(path.name, set())
        source = path.read_text(encoding="utf-8")
        for lib in ("fastapi", "uvicorn", "pywebview"):
            if lib in permitted:
                continue
            assert f"import {lib}" not in source, f"{path.name} imports {lib}"
            assert f"from {lib}" not in source, f"{path.name} imports {lib}"
