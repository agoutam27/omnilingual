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
    """`/` must be the file on disk, and the browser must be able to load it.

    The requests below carry no token, because that is the only shape a browser
    can produce for <link href> and <script src>. Asserting 200 here is what
    proves the page can load its own CSS and JS at all; a version that needed the
    header was a page that came up blank, and this test was what hid it.

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

        # No token header: exactly what the page's own tags send.
        for name in ("style.css", "app.js"):
            served = client.get(f"/static/{name}")
            assert served.status_code == 200, (
                f"the browser cannot load /static/{name} ({served.status_code})")
            assert served.text == _read(name), f"{name} is not the file on disk"
            # The other way round: still a token that guards /api, so the
            # exemption is not the whole token check being dropped.
            assert client.get("/api/audio").status_code == 403
            assert token not in served.text


def test_the_pages_tags_are_plain_links_to_its_own_assets():
    """The page must load the way the brief specifies, not through a bootstrap.

    An inline loader would be a second, untested copy of the asset's loading
    path, and a `<link>`/`<script>` pair is the only shape that needs no token
    header at all.
    """
    html = _read("index.html")
    assert '<link rel="stylesheet" href="/static/style.css">' in html
    assert '<script src="/static/app.js"></script>' in html
    assert "fetch(" not in html, "index.html loads its assets with fetch again"


# --- the page talks to the API the way the server is built -----------------


def test_page_fetches_the_token_and_sends_it_on_every_call():
    js = _read("app.js")
    assert "/token.js" in js, "the token must be loaded, not hard-coded"
    assert "X-Omnilingual-Token" in js
    # The header is sent with the loaded token, not merely named.
    assert re.search(r"\[TOKEN_HEADER\]\s*:\s*token\b", js), (
        "a request goes out without the token in the header")
    # A literal in the page would outlive the launch it was minted for.
    for name in ("index.html", "app.js"):
        assert not re.search(r"token\s*=\s*[\"']", _read(name)), (
            "a token literal is baked into the page")


def test_the_token_is_loaded_by_running_the_script_not_by_reading_its_text():
    """/token.js is JavaScript, and fetch() returns its text without running it.

    A page that only awaits the fetch reads an undefined token and then answers
    every request with a 403, which looks like a dead server rather than a dead
    page. It has to be a <script src>, and the token has to be checked for once
    it has run rather than assumed.
    """
    js = _read("app.js")
    assert 'createElement("script")' in js, "no script element is built"
    assert re.search(r'\bsrc\s*=\s*"/token\.js"', js), (
        "the token script is not loaded by URL, so nothing runs it")
    assert re.search(r"!\s*window\.OMNILINGUAL_TOKEN", js), (
        "the token is used without checking that it arrived")
    # A load failure is an event on the element, not a thrown exception.
    assert "script.onerror" in js or "onerror" in js, (
        "a refused /token.js leaves the window half-loaded and silent")


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


def test_a_cleared_numeric_field_is_sent_as_null_not_zero():
    """Number("") is 0, and 0 is a real budget for the cost cap.

    A cleared cost cap sent as 0 halts the API calls on the first segment and
    reports `halted` with no stated cause; a cleared noise floor changes VAD for
    every chunk; a cleared worker count silently drops to one worker.
    session._numeric documents blank as "not provided", so null is the only
    reading that gets the field's own default back.
    """
    js = _read("app.js")
    helper = _function(js, "numberField")
    assert re.search(r'===\s*""\s*\?\s*null\b', helper), (
        "a cleared numeric field is sent as 0 rather than as 'not provided'")
    assert "Number(" in helper, "a field that is filled must still be a number"
    collect = _function(js, "collect")
    for key in ("num_speakers", "target_s", "max_chunk_s", "min_chunk_s",
                "noise_db", "stt_workers", "max_cost"):
        assert f'numberField("{key}")' in collect, f"{key} bypasses the helper"
        assert f'Number($("{key}")' not in collect, (
            f"{key} turns a cleared box into 0")


def test_page_shows_key_presence_not_key_values():
    js = _read("app.js")
    assert "/api/keys" in js
    # Copy assertion: the backend answers booleans, and these two words are the
    # whole of what the page is allowed to say about a stored key.
    assert "configured" in js and "not configured" in js
    assert re.search(r"\[name\]: value", js), "the write sends no value"
    assert re.search(r"type = \"password\"", js)


def test_a_typed_key_value_reaches_no_node_but_the_one_that_sent_it():
    """The value exists in the password field and in the PUT body, nowhere else.

    Containment rather than a count: every mention of the field's value has to
    be one of the two allowed ones, so a badge that grows the typed value — the
    way a value leaks into a screenshot — fails here.
    """
    js = _read("app.js")
    every = re.findall(r"field\.value", js)
    assert every, "the Save button no longer reads the field"
    allowed = (re.findall(r"writeKey\(name, field\.value", js)
               + re.findall(r"field\.value\s*=\s*\"\"", js))
    assert len(allowed) == len(every), (
        f"field.value is used {len(every)} times but only {len(allowed)} of them "
        "send it or clear it")


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
    # Routing output silently breaks volume keys, so nothing in the page may name
    # the setter or reach for its flag. The flag is banned as a string literal
    # rather than as a bare substring, so a comment mentioning it is not a
    # failure while an actual call still is.
    for name in ("index.html", "style.css", "app.js"):
        body = _read(name)
        assert "SwitchAudioSource" not in body, name
        assert not re.search(r"""["'`]-s["'`]""", body), (
            f"{name} passes SwitchAudioSource -s")


def test_a_wrong_output_device_is_its_own_warning():
    """The backend appends it to detail and keeps it out of `ok`.

    So a ready capture with a wrong output must be a *warning*: painting it as a
    failure would tell the user their audio is broken when the capture works and
    only the speaker route is wrong.
    """
    js = _read("app.js")
    apply = _function(js, "applyAudio")
    assert re.search(r"detail\.length", apply), "detail is never inspected"
    assert re.search(r'ready\.ok\s*\?\s*"warn"\s*:\s*"error"', apply), (
        "the banner level is not taken from the readiness verdict")


def test_the_end_panel_never_substitutes_an_output_path():
    """`end.out` is null when nothing was written, and null is the whole answer.

    A default name here would tell the user a file exists that does not, which is
    the one thing an end panel exists to settle.
    """
    js = _read("app.js")
    end = _function(js, "renderEnd")
    assert "message.out" in end
    assert not re.search(r"message\.out\s*\|\|", end), (
        "the end panel substitutes a path the server did not send")


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
