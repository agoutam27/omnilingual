"""SessionRunner: one run, one daemon thread, one stream of event dicts.

Every test here runs with no audio hardware, no network and no API key. The
live path drives the pipeline through the scripted-capture seam already proven
in tests/pipeline/test_run_live.py, and the batch path through respx plus a
generated WAV.
"""

import json

import pytest
import respx

from omnilingual.config import ConfigError, load_settings
from omnilingual.models import Chunk, Segment
from omnilingual.pipeline.live import LiveOptions
from omnilingual.ui import secrets
from omnilingual.ui.session import SessionRunner, build_run, live_options, run_env

from tests.conftest import make_wav
from tests.pipeline.test_run_live import (  # reuse the proven seams
    FakeCapture,
    _blocks,
    _gaps,
    _pcm_3x20,
    _route,
)


class Collector:
    """An asyncio.Queue stand-in so the runner needs no event loop in tests."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def put_nowait(self, message: dict) -> None:
        self.items.append(message)

    def of(self, kind: str) -> list[dict]:
        return [m for m in self.items if m["type"] == kind]


def _providers():
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    config = load_settings(api_key="k")
    return config, SarvamSTT(config), MayuraTranslator(config)


def _live(tmp_path, capture_factory, **kw):
    queue = Collector()
    runner = SessionRunner(run_id="r1", queue=queue)
    opts = LiveOptions(out=tmp_path / "meeting.md", stt_workers=1, **kw)
    return runner, queue, opts, capture_factory


def _empty_env(tmp_path, monkeypatch):
    """Point the .env at an empty file so a developer's real keys stay out."""
    path = tmp_path / ".env"
    path.write_text("", encoding="utf-8")
    monkeypatch.setenv("OMNILINGUAL_ENV_FILE", str(path))
    return path


# --- the live path --------------------------------------------------------


@respx.mock
def test_live_run_emits_state_segment_status_and_end(respx_mock, tmp_path):
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    config, stt, mt = _providers()
    runner, queue, opts, factory = _live(tmp_path, factory)

    runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                      opts=opts, capture_factory=factory)
    runner.join(timeout=120)

    assert runner.status == "done"
    kinds = [m["type"] for m in queue.items]
    assert kinds[0] == "state", "the page must learn the run exists before any row"
    assert "segment" in kinds
    assert "status" in kinds, "status lines must still reach the page"
    assert kinds[-1] == "end"

    seqs = [m["seq"] for m in queue.of("segment")]
    assert seqs == sorted(seqs), "segments must stream in transcript order"
    for message in queue.of("segment"):
        assert set(message) >= {"seq", "idx", "start_s", "end_s", "lang", "prob",
                                "text", "english", "status", "speaker", "kept",
                                "cost_inr"}
        assert isinstance(message["kept"], bool)

    end = queue.of("end")[0]
    assert end["exit_code"] == 0
    assert end["out"] == str(tmp_path / "meeting.md")
    assert end["recoverable"] is True
    assert end["session_dir"] is not None
    assert (tmp_path / ".omnilingual").is_dir()


@respx.mock
def test_live_run_reports_a_session_dir_for_recovery(respx_mock, tmp_path):
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    config, stt, mt = _providers()
    runner, queue, opts, factory = _live(tmp_path, factory)

    runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                      opts=opts, capture_factory=factory)
    runner.join(timeout=120)

    assert runner.session_dir is not None
    assert (runner.session_dir / "session.json").is_file()
    assert runner.recoverable is True
    assert runner.out_path == tmp_path / "meeting.md"
    end = queue.of("end")[0]
    assert end["session_dir"] == str(runner.session_dir)


@respx.mock
def test_stop_leaves_a_finalized_file(respx_mock, tmp_path):
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    config, stt, mt = _providers()
    runner, queue, opts, factory = _live(tmp_path, factory)

    runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                      opts=opts, capture_factory=factory)
    runner.stop()
    runner.join(timeout=120)

    assert runner.status in {"stopping", "done"}
    assert (tmp_path / "meeting.md").exists()
    assert "· growing" not in (tmp_path / "meeting.md").read_text(encoding="utf-8")
    assert queue.of("end"), "a stopped run still reports how it ended"


@respx.mock
def test_run_thread_is_a_daemon(respx_mock, tmp_path):
    """A daemon thread cannot hold the process open, so quitting the app mid-run
    never blocks on a meeting that is still capturing."""
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    config, stt, mt = _providers()
    runner, queue, opts, factory = _live(tmp_path, factory)

    runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                      opts=opts, capture_factory=factory)
    runner.join(timeout=120)

    assert runner._thread is not None
    assert runner._thread.daemon is True


@respx.mock
def test_a_dead_queue_cannot_kill_a_running_run(respx_mock, tmp_path):
    """Constraint: the run lives in the server process, not in the page. A closed
    event loop or a torn-down WebSocket must cost the UI its updates and nothing
    else — so the guard has to be in the delivery path, because run_live calls the
    status callback without one of its own."""
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    config, stt, mt = _providers()

    class DeadQueue:
        def put_nowait(self, message):
            raise RuntimeError("event loop is closed")

    runner = SessionRunner(run_id="r-dead", queue=DeadQueue())
    opts = LiveOptions(out=tmp_path / "meeting.md", stt_workers=1)
    runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                      opts=opts, capture_factory=factory)
    runner.join(timeout=120)

    assert runner.status == "done"
    assert "Bravo" in (tmp_path / "meeting.md").read_text(encoding="utf-8")
    assert runner.session_dir is not None


@respx.mock
def test_halted_run_reports_halted_and_stays_recoverable(respx_mock, tmp_path):
    """A cost cap stops the API calls but capture continues and the sealed chunks
    stay on disk. That is not a failure: the file is valid and the session is
    recoverable, so 'failed' would be a lie the UI would act on."""
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    config, stt, mt = _providers()
    runner, queue, opts, factory = _live(tmp_path, factory, max_cost=0.0001)

    runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                      opts=opts, capture_factory=factory)
    runner.join(timeout=120)

    assert runner.status == "halted"
    end = queue.of("end")[0]
    assert end["exit_code"] == 2
    assert end["recoverable"] is True
    assert runner.session_dir is not None
    assert "## Transcript" in (tmp_path / "meeting.md").read_text(encoding="utf-8")
    # Every sealed chunk is still told about, halt markers included, so the live
    # view never freezes on a run that has stopped paying.
    assert [m["status"] for m in queue.of("segment")].count("stt_failed") >= 1


# --- the batch paths ------------------------------------------------------


@respx.mock
def test_recording_run_writes_a_rendered_transcript(respx_mock, tmp_path):
    _route(respx_mock)
    recording = tmp_path / "meeting.wav"
    # ("tone", seconds) — raw_pcm has no other part kind.
    make_wav(recording, [("tone", 3.0)])

    config, stt, mt = _providers()
    queue = Collector()
    runner = SessionRunner(run_id="r2", queue=queue)
    out = tmp_path / "ui.md"

    runner.start_recording(settings=config, stt=stt, translator=mt, diarizer=None,
                           source=recording, work_root=tmp_path / "work", out=out)
    runner.join(timeout=300)

    assert runner.status == "done"
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "Transcript" in text
    assert queue.of("segment"), "batch progress must stream as segment messages"
    assert all(m["kept"] is True for m in queue.of("segment"))
    assert queue.of("end")[0]["exit_code"] == 0
    # No live session to come back to: a batch run is re-run, not recovered.
    assert queue.of("end")[0]["recoverable"] is False


@respx.mock
def test_recording_file_is_byte_identical_to_the_cli_renderer(respx_mock, tmp_path):
    """The UI renders nothing of its own, or a UI-written transcript would drift
    from a CLI-written one for the same transcript. One fixed STT answer, so both
    runs transcribe the same thing rather than consuming a queued reply each."""
    import httpx
    from omnilingual.cache import JsonCache
    from omnilingual.pipeline import run as batch_run
    from omnilingual.render.markdown import render
    from tests.pipeline.test_run_live import MT_URL, STT_URL

    respx_mock.post(STT_URL).mock(return_value=httpx.Response(200, json={
        "request_id": "r1", "transcript": "Bravo",
        "language_code": "hi-IN", "language_probability": 0.9}))
    respx_mock.post(MT_URL).mock(return_value=httpx.Response(200, json={
        "translated_text": "Bravo in English"}))

    recording = tmp_path / "meeting.wav"
    make_wav(recording, [("tone", 3.0)])
    config, stt, mt = _providers()

    reference = render(batch_run(recording, tmp_path / "ref", config, stt, mt,
                                 JsonCache(tmp_path / "ref" / "cache")))
    queue = Collector()
    runner = SessionRunner(run_id="r2b", queue=queue)
    out = tmp_path / "ui.md"
    runner.start_recording(settings=config, stt=stt, translator=mt, diarizer=None,
                           source=recording, work_root=tmp_path / "work", out=out)
    runner.join(timeout=300)

    assert runner.status == "done"
    assert out.read_text(encoding="utf-8") == reference


@respx.mock
def test_recover_finishes_a_halted_live_session(respx_mock, tmp_path):
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    config, stt, mt = _providers()
    runner, queue, opts, factory = _live(tmp_path, factory, max_cost=0.0001)
    runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                      opts=opts, capture_factory=factory)
    runner.join(timeout=120)
    session_dir = runner.session_dir
    assert session_dir is not None

    # A second, independent runner: recovery is its own run, not a replay of the
    # first one's state.
    queue2 = Collector()
    recovered = SessionRunner(run_id="r4", queue=queue2)
    out = tmp_path / "recovered.md"
    recovered.recover(session_dir=session_dir, settings=config, stt=stt,
                      translator=mt, diarizer=None, out=out)
    recovered.join(timeout=300)

    assert recovered.status == "done"
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "Bravo" in text, "the halted run's sealed chunks must now be transcribed"
    assert queue2.of("end")[0]["exit_code"] == 0
    assert recovered.session_dir == session_dir


def test_recording_failure_is_reported_not_raised(tmp_path):
    config, stt, mt = _providers()
    queue = Collector()
    runner = SessionRunner(run_id="r3", queue=queue)

    runner.start_recording(settings=config, stt=stt, translator=mt, diarizer=None,
                           source=tmp_path / "nope.wav",
                           work_root=tmp_path / "work", out=tmp_path / "x.md")
    runner.join(timeout=60)

    assert runner.status == "failed"
    logs = queue.of("log")
    assert logs, "a failure must produce a log message, not silence"
    assert "nope.wav" in " ".join(m["message"] for m in logs)
    assert queue.of("end")[0]["exit_code"] != 0
    state = queue.of("state")[-1]
    assert state["error"], "the final state must carry why the run failed"


def test_a_runner_cannot_be_started_twice(tmp_path):
    """The server's registry enforces one run at a time; the runner still refuses
    to be reused, so a bug above it cannot interleave two transcripts into one
    event stream. The capture is scripted, so nothing here reaches for hardware."""
    config, stt, mt = _providers()
    runner = SessionRunner(run_id="r5", queue=Collector())
    opts = LiveOptions(out=tmp_path / "meeting.md", stt_workers=1)
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=[], gated=[])
    runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                      opts=opts, capture_factory=factory)
    runner.join(timeout=30)
    first_thread = runner._thread
    with pytest.raises(RuntimeError, match="already been started"):
        runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                          opts=opts, capture_factory=factory)
    assert runner._thread is first_thread, "a refused restart must spawn no thread"


# --- panel values -> run objects ------------------------------------------


def test_live_options_maps_the_panel_and_keeps_the_work_dir_trap(tmp_path):
    """settings.py stores work_dir as "" on purpose: an empty value must reach
    LiveOptions as None so the pipeline's own default applies. A stored
    ".omnilingual" would instead pin the work tree to the server's directory."""
    out = tmp_path / "standup.md"

    default = live_options({}, out=out)
    assert default.work_root is None
    assert default.device == "Omnilingual"
    assert default.english_only is False

    stored = live_options({"work_dir": str(tmp_path / "cache")}, out=out)
    assert stored.work_root == tmp_path / "cache"

    override = live_options({"work_dir": "/ignored"}, out=out,
                            work_root=tmp_path / "explicit")
    assert override.work_root == tmp_path / "explicit"


def test_live_options_mirrors_the_cli_flag_defaults(tmp_path):
    opts = live_options({"device": "Chat", "mic_only": True, "target_s": 7.0,
                         "max_chunk_s": 20.0, "min_chunk_s": 4.0,
                         "noise_db": -40.0, "stt_workers": 3, "max_cost": 12.5,
                         "english_only": True}, out=tmp_path / "x.md")
    assert (opts.device, opts.mic_only, opts.english_only) == ("Chat", True, True)
    assert (opts.target_s, opts.max_chunk_s, opts.min_chunk_s) == (7.0, 20.0, 4.0)
    assert (opts.noise_db, opts.stt_workers, opts.max_cost) == (-40.0, 3, 12.5)


def test_build_run_treats_an_empty_model_as_the_provider_default(tmp_path, monkeypatch):
    """The panel stores "" for "provider default", which Settings spells None.
    Passing "" through would pin today's default model id into the run, so flipping
    --stt would drag a Sarvam model along behind it."""
    _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "k")

    settings, *_ = build_run({"mode": "live", "stt": "sarvam", "mt": "mayura",
                              "stt_model": "", "mt_model": ""})
    assert settings.stt_model is None
    assert settings.resolved_stt_model == "saaras:v4"
    assert settings.mt_model is None
    assert settings.resolved_mt_model == "mayura:v1"


def test_build_run_maps_the_speaker_toggle_the_way_the_cli_does(tmp_path, monkeypatch):
    import importlib.util

    _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "k")

    off, *_ = build_run({"mode": "live", "stt": "sarvam", "mt": "mayura"})
    assert off.diarizer is None

    panel = {"mode": "live", "stt": "sarvam", "mt": "mayura", "diarize": True,
             "num_speakers": 3}
    if importlib.util.find_spec("sherpa_onnx") is None:
        # Without the extra, a missing model is a ConfigError the panel turns into
        # a 400 carrying the fix — the same contract the CLI gives, and the reason
        # the test passes either way rather than being skipped.
        with pytest.raises(ConfigError, match="extra"):
            build_run(panel)
        return
    on, _, _, diarizer = build_run(panel)
    assert on.diarizer == "sherpa"
    assert on.num_speakers == 3
    assert diarizer is not None


# --- the event shapes -----------------------------------------------------


def test_segment_message_carries_every_field_the_page_renders():
    seg = Segment(chunk=Chunk(idx=3, start_s=1.5, end_s=9.5, wav_path="x.wav"),
                  lang="hi-IN", prob=0.91, text="नमस्ते", english="Hello",
                  status="ok", speaker="S1")
    assert SessionRunner.segment_message(1, seg, 0.08, True) == {
        "type": "segment", "seq": 1, "idx": 3, "start_s": 1.5, "end_s": 9.5,
        "lang": "hi-IN", "prob": 0.91, "text": "नमस्ते", "english": "Hello",
        "status": "ok", "speaker": "S1", "kept": True, "cost_inr": 0.08,
    }


def test_english_is_none_for_silence():
    seg = Segment(chunk=Chunk(idx=0, start_s=0.0, end_s=4.0, wav_path="x.wav"),
                  lang="unknown", prob=0.0, text="", english=None,
                  status="no_speech")
    assert SessionRunner.segment_message(0, seg, 0.0, False)["english"] is None


def test_snapshot_is_the_state_payload_the_spec_documents(tmp_path):
    runner = SessionRunner(run_id="r6", queue=Collector())
    state = runner.snapshot()
    assert state["run_id"] == "r6"
    assert state["status"] == "idle"
    assert state["mode"] == "live"
    for key in ("elapsed_s", "cost_inr", "cost_cap", "segments", "dropped", "out",
                "session_dir", "error"):
        assert key in state, key
    # No live cap on a batch run: reporting ₹0 would read as "budget exhausted".
    assert state["cost_cap"] is None
    assert state["out"] is None


# --- key sourcing: constraint 2 -------------------------------------------


def test_run_env_reads_the_env_file_for_keys_the_process_lacks(tmp_path, monkeypatch):
    path = _empty_env(tmp_path, monkeypatch)
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    path.write_text("SARVAM_API_KEY=from-file\n", encoding="utf-8")

    assert run_env()["SARVAM_API_KEY"] == "from-file"


def test_run_env_lets_the_process_environment_win(tmp_path, monkeypatch):
    # `uv run --env-file .env` puts the key in the environment already, and an
    # explicit export is a more deliberate act than a line in a file.
    path = _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "from-process")
    path.write_text("SARVAM_API_KEY=from-file\n", encoding="utf-8")

    assert run_env()["SARVAM_API_KEY"] == "from-process"


@pytest.mark.parametrize("body,expected", [
    # Last line wins, matching present()'s tail -n 1.
    ("SARVAM_API_KEY=first\nSARVAM_API_KEY=second\n", "second"),
    # `export NAME=` is the key's line, as present() and env_set both treat it.
    ("export SARVAM_API_KEY=exported\n", "exported"),
    # A dotenv loader strips matching quotes.
    ("SARVAM_API_KEY=\"quoted\"\n", "quoted"),
    ("SARVAM_API_KEY='single'\n", "single"),
    # Whitespace around the value is not part of it.
    ("SARVAM_API_KEY =  spaced  \n", "spaced"),
    # A later empty assignment wins, which is what `export K=` then `K=v` does.
    ("SARVAM_API_KEY=v\nSARVAM_API_KEY=\n", None),
])
def test_run_env_resolves_keys_the_way_present_does(body, expected, tmp_path, monkeypatch):
    """secrets.py deliberately publishes no reader, so run_env parses the .env
    itself. These cases are the whole seam between the two: if present() and
    run_env() ever disagree, the panel would report 'configured' for a key the run
    cannot see. Duplicated parsing is only tolerable while a test pins the two
    implementations to each other."""
    _empty_env(tmp_path, monkeypatch)
    secrets.env_path().write_text(body, encoding="utf-8")
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)

    resolved = run_env().get("SARVAM_API_KEY")
    assert resolved == expected
    assert bool(secrets.present()["SARVAM_API_KEY"]) is bool(resolved)


def test_run_env_agrees_with_present_on_an_undecodable_env_file(tmp_path, monkeypatch):
    """secrets.present() reports every key absent for an .env it cannot decode,
    and the file may hold keys this process never read — so run_env contributes
    nothing rather than guessing at the readable part."""
    _empty_env(tmp_path, monkeypatch)
    secrets.env_path().write_bytes(b"SARVAM_API_KEY=ok\n\xff\xfe not utf8\n")
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)

    assert secrets.present() == {"SARVAM_API_KEY": False, "GROQ_API_KEY": False,
                                 "GEMINI_API_KEY": False}
    assert "SARVAM_API_KEY" not in run_env()


def test_run_env_keeps_unrelated_variables_the_loader_would_export(tmp_path, monkeypatch):
    # `uv run --env-file .env` loads the whole file, not just the three keys, and
    # a future provider key must not need a change in two places.
    _empty_env(tmp_path, monkeypatch)
    secrets.env_path().write_text("SOME_FUTURE_KEY=abc\n", encoding="utf-8")

    assert run_env()["SOME_FUTURE_KEY"] == "abc"


def test_build_run_sends_keys_through_load_settings_env(tmp_path, monkeypatch):
    """The key reaches the provider because it is in the environment the run
    inherits, so it never has to be named in a flag, an argv entry, or a message."""
    _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "secret-value")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    settings, stt, translator, diarizer = build_run({
        "mode": "live", "stt": "sarvam", "mt": "mayura",
        "langs": [], "target_s": 8.0, "max_chunk_s": 28.0, "min_chunk_s": 5.0,
    })

    assert settings.api_key == "secret-value"
    assert settings.stt_provider == "sarvam"
    assert settings.mt_provider == "mayura"
    assert diarizer is None
    assert stt is not None and translator is not None


@respx.mock
def test_no_event_ever_carries_a_key_value(respx_mock, tmp_path, monkeypatch):
    _route(respx_mock)
    monkeypatch.setenv("SARVAM_API_KEY", "sk-do-not-log-me")
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    settings = load_settings(env=run_env(), langs=[])
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    queue = Collector()
    runner = SessionRunner(run_id="r7", queue=queue)
    opts = LiveOptions(out=tmp_path / "meeting.md", stt_workers=1)
    runner.start_live(settings=settings, stt=SarvamSTT(settings),
                      translator=MayuraTranslator(settings), diarizer=None,
                      opts=opts, capture_factory=factory)
    runner.join(timeout=120)

    assert runner.status == "done", "the sentinel key really was in play"
    blob = json.dumps(queue.items, default=str)
    assert "sk-do-not-log-me" not in blob


# --- validation is delegated, never copied --------------------------------


@pytest.mark.parametrize("values", [
    {"max_chunk_s": 40.0, "min_chunk_s": 5.0},          # over the 30 s endpoint cap
    {"max_chunk_s": 6.0, "min_chunk_s": 5.0},           # less than 2x min
    {"max_chunk_s": 28.0, "min_chunk_s": 30.0},         # min above max
])
def test_build_run_defers_chunk_bounds_to_the_shared_validator(values, tmp_path, monkeypatch):
    _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "k")
    panel = {"mode": "live", "stt": "sarvam", "mt": "mayura", "langs": [],
             "target_s": 8.0, "max_chunk_s": 28.0, "min_chunk_s": 5.0}
    panel.update(values)
    with pytest.raises(ConfigError):
        build_run(panel)


def test_build_run_defers_target_s_to_the_shared_validator(tmp_path, monkeypatch):
    _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "k")
    panel = {"mode": "live", "stt": "sarvam", "mt": "mayura", "langs": [],
             "max_chunk_s": 28.0, "min_chunk_s": 5.0, "target_s": 60.0}
    with pytest.raises(ConfigError):
        build_run(panel)


def test_build_run_defers_provider_names_to_load_settings(tmp_path, monkeypatch):
    _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "k")
    panel = {"mode": "live", "stt": "not-a-backend", "mt": "mayura", "langs": [],
             "target_s": 8.0, "max_chunk_s": 28.0, "min_chunk_s": 5.0}
    with pytest.raises(ConfigError):
        build_run(panel)


def test_build_run_rejects_a_panel_field_the_cli_has_no_flag_for():
    # The parameter surface is exactly the CLI's; an invented option is a bug, and
    # silently dropping one is how a renamed control becomes a setting that does
    # nothing.
    with pytest.raises(ValueError):
        build_run({"mode": "live", "stt": "sarvam", "mt": "mayura", "turbo": True})
