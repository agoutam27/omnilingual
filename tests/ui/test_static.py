"""The page: three files, and what they are forbidden to claim about the backend.

`app.js` is a thin renderer, so there is no JS runner in this repo and none may
be added. What can be asserted from Python is structural: which ids the panel
wires up, which of the server's message types it handles, which run states it
names, and — the assertions that matter most here — which fields it is *not*
allowed to read. Every test below names a behaviour that `ui/server.py`,
`ui/session.py` or the task's constraints pin down, and each one fails if that
behaviour is taken out.

Copy assertions exist where the behaviour *is* a word ("configured", "not
checked", "unknown"); they are marked in the test names or bodies.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from omnilingual.ui import settings

STATIC = Path(__file__).resolve().parents[2] / "omnilingual" / "ui" / "static"
HOST = "127.0.0.1:5599"

# The six run states session.SessionRunner can report. Every one needs its own
# rendering: 'done' in particular is not success.
RUN_STATES = ("idle", "running", "stopping", "done", "failed", "halted")


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _function(js: str, name: str) -> str:
    """The body of a top-level `function name(...)`, up to its closing brace."""
    start = js.index(f"function {name}(")
    return js[start:js.index("\n}", start)]


# --- the files exist and are the ones the server serves --------------------


def test_all_three_assets_exist():
    for name in ("index.html", "style.css", "app.js"):
        assert (STATIC / name).is_file(), f"{name} missing"


def test_the_page_serves_the_real_assets_not_the_placeholder():
    """`/` must be the file on disk, and both assets must be reachable.

    The static mount is created when the directory exists, so a renamed or
    missing asset shows up here as the placeholder page or a 404 rather than as
    a silently unstyled window.
    """
    pytest.importorskip("fastapi", reason="the ui extra is not installed")
    from fastapi.testclient import TestClient

    from omnilingual.ui.server import create_app, mint_token

    token = mint_token()
    with TestClient(create_app(token=token, port=5599),
                    base_url=f"http://{HOST}") as client:
        page = client.get("/")
        assert page.status_code == 200
        assert page.text == _read("index.html"), "the placeholder page is served"

        # /static is token-guarded like every other path, so the page's own
        # files arrive with the header and are refused without it.
        for name in ("style.css", "app.js"):
            served = client.get(f"/static/{name}",
                                headers={"X-Omnilingual-Token": token})
            assert served.status_code == 200, name
            assert served.text == _read(name), f"{name} is not the file on disk"
            refused = client.get(f"/static/{name}")
            assert refused.status_code == 403, f"/static/{name} is unguarded"


# --- the page talks to the API the way the server is built -----------------


def test_page_fetches_the_token_and_sends_it_on_every_call():
    js = _read("app.js")
    assert "/token.js" in js, "the token must be fetched, not hard-coded"
    assert "X-Omnilingual-Token" in js
    # The header is sent with the fetched token, not merely named: every call
    # goes through one helper, and the bootstrap that loads the two guarded
    # assets sends it too.
    for body in (js, _read("index.html")):
        assert re.search(r"\[TOKEN_HEADER\]\s*:\s*token\b", body), (
            "a request goes out without the token in the header")
    assert js.count("await fetch(") == 2, "requests must go through api()"
    html = _read("index.html")
    assert re.search(r"fetch\(\"/token\.js\"\)", html), (
        "the bootstrap that loads the assets never fetches the token")
    # A literal in the page would outlive the launch it was minted for.
    assert not re.search(r"token\s*=\s*[\"']", html), (
        "a token literal is baked into the page")


def test_the_token_script_is_executed_not_merely_fetched():
    """/token.js is JavaScript, and fetch() returns its text without running it.

    A page that only awaits the fetch reads an undefined token and then answers
    every request with a 403, which looks like a dead server rather than a dead
    page. Both the bootstrap and app.js must put the response text into a script
    element, which is what executes it.
    """
    for name in ("index.html", "app.js"):
        body = _read(name)
        assert 'createElement("script")' in body, f"{name} builds no script element"
        assert re.search(
            r'\w+\.textContent\s*=\s*await\s*\(\s*await fetch\("/token\.js"\)\)\.text\(\)',
            body), f"{name} fetches the token script without executing it"


def test_every_panel_field_is_loaded_from_the_stored_defaults():
    """The three load lists plus the four specially-loaded fields span the store.

    A field that falls out of all of them keeps whatever the browser last had in
    the box, which is the one failure a user cannot tell from a stale value.
    """
    js = _read("app.js")
    lists = {}
    for name in ("TEXT_FIELDS", "NUMBER_FIELDS", "CHECK_FIELDS"):
        block = re.search(rf"const {name} = \[(.*?)\];", js, re.S)
        assert block, f"{name} is gone from app.js"
        lists[name] = set(re.findall(r'"([a-z_]+)"', block.group(1)))
    # mode, langs, stt and mt are loaded by their own lines in loadDefaults.
    covered = set().union(*lists.values()) | {"mode", "langs", "stt", "mt"}
    assert covered == set(settings.DEFAULTS), (
        f"never loaded: {sorted(set(settings.DEFAULTS) - covered)}; "
        f"not a field: {sorted(covered - set(settings.DEFAULTS))}")
    assert not lists["TEXT_FIELDS"] & lists["NUMBER_FIELDS"], (
        "a field is loaded as two different types")


def test_every_element_the_script_reaches_for_exists_in_the_page():
    """A rename on one side only is a null dereference at run time."""
    html = _read("index.html")
    reached = set(re.findall(r"\$\(\"([A-Za-z_-]+)\"\)", _read("app.js")))
    reached |= set(re.findall(r"getElementById\(\"([A-Za-z_-]+)\"\)", html))
    assert reached, "no ids were looked up at all"
    for name in sorted(reached):
        assert f'id="{name}"' in html, f"#{name} is read but not in the page"


def test_page_opens_a_websocket_with_the_token():
    js = _read("app.js")
    assert re.search(r"/ws\b", js)
    assert "WebSocket" in js
    assert "token=" in js


def test_page_covers_both_modes_and_every_parameter_control():
    """Every settings.DEFAULTS key is a control, and the mode is a radio pair.

    `mode` is the one field with no id: it is a group of two radios, so the
    assertion is on the group. Every other field is addressed by id in both the
    HTML and app.js, which is what lets the panel round-trip ui.toml.
    """
    html = _read("index.html")
    js = _read("app.js")
    for key in settings.DEFAULTS:
        if key == "mode":
            continue
        assert f'id="{key}"' in html, f"panel control {key} missing"
    assert 'name="mode"' in html
    for mode in ("live", "recording"):
        assert f'name="mode" value="{mode}"' in html
    # A control the script never reads is a control that silently does nothing.
    for key in settings.DEFAULTS:
        if key == "mode":
            assert "input[name=mode]" in js, "the mode radios are never read"
            continue
        assert f'"{key}"' in js or f"${key}" in js, f"{key} is never read"


def test_the_start_payload_is_exactly_the_panel_the_server_accepts():
    """session.build_run rejects a field settings.DEFAULTS does not name.

    So the payload is checked against the store itself rather than a list
    copied into this file: an invented key is a 400 on every start, and a
    dropped key is a setting the user cannot change.
    """
    js = _read("app.js")
    block = _function(js, "collect")
    sent = set(re.findall(r"^\s*([a-z_]+):", block, re.M))
    assert sent == set(settings.DEFAULTS), (
        f"start payload is {sorted(sent)}, server accepts "
        f"{sorted(settings.DEFAULTS)}")


def test_page_renders_every_server_message_type():
    js = _read("app.js")
    for kind in ("state", "segment", "status", "log", "end"):
        assert f'case "{kind}":' in js, f"no handler for the {kind} message"


def test_all_six_run_states_have_their_own_rendering():
    js = _read("app.js")
    labels = re.search(r"const RUN_STATE = \{(.*?)\n\};", js, re.S)
    assert labels, "no per-state labels"
    for state in RUN_STATES:
        assert re.search(rf"^\s*{state}:", labels.group(1), re.M), (
            f"{state} shares another state's rendering")
    # 'stopping' holds the window; 'halted' and 'done' do not.
    busy = re.search(r"const BUSY = .*", js)
    assert busy and "running" in busy.group(0) and "stopping" in busy.group(0)
    # The state is written onto the DOM, so the six are distinguishable and not
    # only six different strings in a table.
    assert "dataset.status" in js


def test_done_is_not_rendered_as_unqualified_success():
    """`done` also covers exit_code 2, which is a written file with failures."""
    js = _read("app.js")
    end = _function(js, "renderEnd")
    assert "exit_code" in end, "the end message's exit code is ignored"
    assert re.search(r"exit_code === 0", end)
    assert re.search(r"exit_code === 2", end)


def test_stop_is_offered_only_for_a_run_that_can_stop():
    """Recording and recovery runs have no stop channel at all."""
    js = _read("app.js")
    gate = _function(js, "canStop")
    assert 'mode === "live"' in gate, "stop must not be offered for a batch run"
    assert "BUSY.has(snapshot.status)" in gate
    assert 'onclick = stop' in js


def test_no_live_cost_meter_is_built_from_the_always_zero_field():
    """state.cost_inr and segment.cost_inr are 0.0 for every live run.

    cost_cap is real, so it is shown and labelled as a cap; the field that can
    only ever be zero must not be read at all.
    """
    for name in ("index.html", "style.css", "app.js"):
        # Prose may name the field; a property read or a key lookup may not.
        assert not re.search(r"[.\"'\[]cost_inr", _read(name)), \
            f"{name} reads cost_inr"
    js = _read("app.js")
    assert "cost_cap" in js
    assert re.search(r"cost cap", js, re.I), "the cap must be labelled as a cap"


def test_page_shows_key_presence_not_key_values():
    js = _read("app.js")
    assert "/api/keys" in js
    # Copy assertion: the backend answers booleans, and these two words are the
    # whole of what the page is allowed to say about a stored key.
    assert "configured" in js and "not configured" in js
    # The typed value exists only in the password field and the PUT body.
    assert re.search(r"\[name\]: value", js) or re.search(r"\[name\]: field\.value", js)
    assert re.search(r"field\.value\s*=\s*\"\"", js), (
        "the typed value stays in the DOM after a successful write")
    assert re.search(r"type = \"password\"", js)


def test_a_refused_key_write_is_reported_and_never_retried():
    """PUT /api/keys answers 400 when the .env cannot be decoded."""
    js = _read("app.js")
    write = _function(js, "writeKey")
    assert "showBanner" in write
    assert not re.search(r"\bwhile\b|\bfor\b", write), "a key write must not retry"
    assert "loadKeys" in write, "the badges are re-read after a write"


def test_page_offers_recovery_only_when_the_run_is_recoverable():
    js = _read("app.js")
    assert "recoverable" in js
    # Both places that offer it — the run that just ended, and the catalogue
    # read back from disk — must ask first. A Recover that always fails is worse
    # than none: run_from_chunks refuses a session with no sealed chunk.
    for where in ("renderEnd", "loadRuns"):
        assert re.search(r"if\s*\([^)]*recoverable[^)]*\)", _function(js, where)), (
            f"{where} offers recovery without asking whether it can succeed")


def test_page_never_switches_the_system_output_itself():
    # Routing output silently breaks volume keys; only the server may report it.
    for name in ("index.html", "style.css", "app.js"):
        body = _read(name)
        assert "SwitchAudioSource" not in body, name
        assert "-s " not in body, name


def test_a_wrong_output_device_is_its_own_warning():
    """The backend appends it to detail and keeps it out of `ok`."""
    js = _read("app.js")
    apply = _function(js, "applyAudio")
    assert re.search(r"detail\.length", apply), "detail is never inspected"
    # ok drives the severity, so a warning the backend considers non-fatal is
    # not painted like a failure.
    assert "ready.ok" in apply


def test_the_microphone_is_never_reported_as_authorised_when_it_was_not_probed():
    js = _read("app.js")
    assert "mic_checked" in js, "a skipped probe would read as a working mic"
    # The polarity is the behaviour: only a probe that did not run says "not
    # checked", so the negation has to be on the checked flag itself.
    assert re.search(r"!\s*ready\.mic_checked\s*\?\s*\"not checked\"", js)
    # Copy assertion.
    assert "not checked" in js


def test_a_validation_failure_shows_the_servers_own_sentence():
    """Every 400 is one ConfigError, so the string is the whole answer."""
    js = _read("app.js")
    assert re.search(r"body\.error", js)
    assert not re.search(r"\balert\(|\bconfirm\(", js)


def test_a_second_run_is_reported_rather_than_crashed():
    """POST /api/session/start answers 409 while a run is active."""
    js = _read("app.js")
    assert "409" in js
    start = _function(js, "start")
    assert "showBanner" in start
    assert "loadRuns" in start, "a refused start does not re-read the catalogue"


def test_the_three_detached_routes_are_never_treated_as_synchronous():
    """They answer 200 {"started": true} and report on the log channel."""
    js = _read("app.js")
    for path in ("/api/audio/setup", "/api/audio/restart-daemon",
                 "/api/setup/apply"):
        # Quoted, so a renamed key cannot satisfy a substring match.
        assert f'"{path}"' in js, f"{path} is not reached"
    detached = _function(js, "detached")
    assert re.search(r"answer\.started", detached), (
        "the acknowledgement is not what tells the page work began")
    repoll = _function(js, "repollAudio")
    assert '"/api/audio"' in repoll, "readiness is never re-read after the work"
    assert "setTimeout" in repoll, "the re-poll busy-waits"


def test_a_closed_window_does_not_take_the_run_with_it():
    """Reconnect, then re-read the catalogue and re-render from the handshake."""
    js = _read("app.js")
    assert "/api/runs" in js
    assert "onclose" in js
    assert re.search(r"setTimeout\(\s*connect", js), "no reconnect"
    assert "loadRuns()" in _function(js, "connect"), (
        "a reconnected window does not re-read the catalogue")
    assert not re.search(r"while\s*\(true\)|setInterval", js), "a busy-wait loop"


def test_a_session_found_on_disk_shows_an_unknown_output_path():
    """GET /api/runs reports out: null for a past session, honestly."""
    js = _read("app.js")
    runs = _function(js, "loadRuns")
    assert re.search(r"entry\.out\s*\|\|", runs), "a path is synthesised"
    # Copy assertion: the fallback word is the behaviour — null means the
    # manifest does not record where the transcript went.
    assert '"output path unknown"' in runs


def test_nothing_is_ever_injected_as_markup():
    """textContent only: a transcript or a log line is never parsed as HTML."""
    for name in ("index.html", "style.css", "app.js"):
        js = _read(name)
        cleared = set(re.findall(r"innerHTML\s*=\s*(\"[^\"]*\"|'')", js))
        assert cleared <= {'""', "''"}, f"{name} assigns markup: {cleared}"
        assert len(re.findall(r"innerHTML", js)) == len(cleared), (
            f"{name} assigns something other than an empty string to innerHTML")


def test_the_page_loads_nothing_from_off_the_machine():
    """Same-origin only: no CDN, no web font, no analytics."""
    html = _read("index.html")
    assert not re.search(r'(?:src|href)\s*=\s*"https?:', html)
    assert "/static/style.css" in html or "/static/app.js" in html
    js = _read("app.js")
    assert "https://" not in js and "http://" not in js
    assert "@import" not in _read("style.css")
