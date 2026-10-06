"""SessionRunner: one run, one daemon thread, one stream of event dicts.

Every test here runs with no audio hardware, no network and no API key. The
live path drives the pipeline through the scripted-capture seam already proven
in tests/pipeline/test_run_live.py, and the batch path through respx plus a
generated WAV.
"""

import asyncio
import json
import threading
import time
from dataclasses import replace

import pytest
import respx

from omnilingual.config import ConfigError, load_settings
from omnilingual.models import Chunk, Segment
from omnilingual.pipeline.live import LiveOptions
from omnilingual.ui import secrets
from omnilingual.ui.session import (
    SessionRunner,
    build_run,
    free_output_path,
    live_options,
    run_env,
)

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


@respx.mock
def test_a_dropped_silence_row_carries_no_english_and_counts_as_dropped(respx_mock, tmp_path):
    """The real contract is which rows carry English and which do not, and which
    ones the writers dropped — not that None passes through a dict. The middle
    chunk's whitespace-only answer becomes a no_speech segment that the file drops
    but the page still has to be told about, or a quiet stretch freezes the view."""
    _route(respx_mock, texts=("Bravo", "   ", "Vanakkam"))
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    config, stt, mt = _providers()
    runner, queue, opts, factory = _live(tmp_path, factory)

    runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                      opts=opts, capture_factory=factory)
    runner.join(timeout=120)

    silence = [m for m in queue.of("segment") if m["status"] == "no_speech"]
    assert silence, "the dropped-silence row is the contract"
    for row in silence:
        assert row["english"] is None
        assert row["kept"] is False
        assert isinstance(row["kept"], bool)

    spoken = [m for m in queue.of("segment") if m["status"] == "ok"]
    assert spoken and all(m["english"] for m in spoken), "translated rows carry English"
    # The dropped counter is what the page shows as "N chunks skipped".
    final = queue.of("state")[-1]
    assert final["dropped"] >= 1
    assert final["segments"] == len(queue.of("segment"))
    # ...and the dropped silence is genuinely absent from the file, or "dropped"
    # would be a label for a row that was written after all.
    assert "no speech detected" not in (tmp_path / "meeting.md").read_text(encoding="utf-8")


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


def test_run_env_lets_the_env_file_beat_the_launch_environment(tmp_path, monkeypatch):
    # Precedence is a deliberate decision, not an accident: the .env is the file
    # the panel manages, so a value exported when the app happened to be launched
    # must not silently override what the user just typed into it.
    path = _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "from-process")
    path.write_text("SARVAM_API_KEY=typed-into-panel\n", encoding="utf-8")

    assert run_env()["SARVAM_API_KEY"] == "typed-into-panel"
    # And the panel's own badge agrees with what the run will do.
    assert secrets.present()["SARVAM_API_KEY"] is True


def test_run_env_still_fills_names_the_env_file_omits(tmp_path, monkeypatch):
    # The launch environment is still the source for everything the .env does not
    # mention — that is what keeps `uv run --env-file .env`, and a
    # `FOO=bar uv run omnilingual-ui`, working.
    path = _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SOME_FUTURE_KEY", "from-process")
    path.write_text("SARVAM_API_KEY=from-file\n", encoding="utf-8")

    resolved = run_env()
    assert resolved["SOME_FUTURE_KEY"] == "from-process"
    assert resolved["SARVAM_API_KEY"] == "from-file"


def test_an_environment_only_key_is_usable_while_the_badge_says_no(tmp_path, monkeypatch):
    """The one intentional asymmetry with secrets.present(), asserted so it stays
    a stated contract: the badge answers "is this key in the file the panel
    manages", and an exported key is a property of how the app was started. The
    run uses it; the badge does not claim it. Harming the user here would be the
    alternative — refusing a key that works."""
    _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "exported-in-the-shell")

    assert secrets.present()["SARVAM_API_KEY"] is False
    assert run_env()["SARVAM_API_KEY"] == "exported-in-the-shell"


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
    ("SARVAM_API_KEY=v\nSARVAM_API_KEY=\n", ""),
])
def test_run_env_resolves_keys_the_way_present_does(body, expected, tmp_path, monkeypatch):
    """secrets.py deliberately publishes no reader, so run_env parses the .env
    itself. These cases are the whole seam between the two. The variable is left
    in os.environ on purpose: that is the state the previous structural pin could
    not see, and it is where the two answers came to disagree.

    The invariant is one-directional and is the one that could bite a user:
    present() reporting a key must mean the run uses that key's .env value, never
    an older exported one. present() reporting a key absent does NOT mean the run
    lacks it — see the environment-only test above."""
    _empty_env(tmp_path, monkeypatch)
    secrets.env_path().write_text(body, encoding="utf-8")
    monkeypatch.setenv("SARVAM_API_KEY", "from-process")

    resolved = run_env().get("SARVAM_API_KEY")
    assert resolved == expected
    assert bool(secrets.present()["SARVAM_API_KEY"]) is bool(expected)


def test_a_blank_final_assignment_exposes_the_environment_not_an_earlier_line(
    tmp_path, monkeypatch
):
    """`K=v` then `K=` is a deletion, not a redefinition — of the whole key, in both
    layers. Reading it as "keep the first value" would resurrect a key the user
    just cleared in the panel, and reading it as "keep the shell's" would let an
    exported key override a .env that says empty. Blank wins, so the panel and the
    run agree: no key."""
    _empty_env(tmp_path, monkeypatch)
    secrets.env_path().write_text("SARVAM_API_KEY=stale\nSARVAM_API_KEY=\n",
                                  encoding="utf-8")
    monkeypatch.setenv("SARVAM_API_KEY", "from-process")

    assert secrets.present()["SARVAM_API_KEY"] is False
    # The .env layer wins even when it wins with an empty value...
    assert run_env()["SARVAM_API_KEY"] == ""
    # ...and load_settings' own truthiness reading turns that into no key at all,
    # which is the agreement that matters: nothing is usable that the panel calls
    # unconfigured.
    assert load_settings(env=run_env()).api_key is None


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
    # nothing. ConfigError, not ValueError: the server answers 400 for it, and a
    # second exception type would turn a renamed control into a 500.
    with pytest.raises(ConfigError, match="unknown run field"):
        build_run({"mode": "live", "stt": "sarvam", "mt": "mayura", "turbo": True})


# --- the whole spec parameter table, one payload, both consumers ------------

# Spec §9's table, field for field. The server receives this and hands it to
# build_run and to live_options unfiltered; a field either rejects it outright.
SPEC_PANEL = {
    "mode": "live", "source": "", "out": "standup.md",
    "stt": "sarvam", "stt_model": "", "mt": "mayura", "mt_model": "",
    "diarize": False, "num_speakers": 3, "langs": [], "english_only": False,
    "work_dir": "", "device": "Omnilingual", "mic_only": False,
    "target_s": 8.0, "max_chunk_s": 28.0, "min_chunk_s": 5.0,
    "noise_db": -35.0, "stt_workers": 2, "max_cost": 50.0,
}


def test_the_whole_spec_panel_is_accepted_by_both_consumers(tmp_path, monkeypatch):
    """Both functions take the same payload. They read disjoint subsets of it, so
    neither may reject a field the other owns — that is the bug that made every
    legitimate live start fail validation."""
    _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "k")

    settings, stt, translator, diarizer = build_run(dict(SPEC_PANEL))
    opts = live_options(dict(SPEC_PANEL), out=tmp_path / "standup.md")

    assert settings.stt_provider == "sarvam"
    assert settings.mt_provider == "mayura"
    assert stt is not None and translator is not None
    assert diarizer is None
    assert opts.device == "Omnilingual"
    assert opts.work_root is None  # work_dir "" -> the pipeline's own default
    assert (opts.stt_workers, opts.max_cost, opts.english_only) == (2, 50.0, False)


def test_the_panel_allowlist_is_the_store_itself():
    # Not a restated copy: a second list drifts, and the drift rejects a start
    # request for a field the server never mentioned.
    from omnilingual.ui import settings as ui_settings

    assert set(SPEC_PANEL) == set(ui_settings.DEFAULTS)


# --- a missing credential can never start a run ---------------------------


@pytest.mark.parametrize("stt,mt,configured,message", [
    ("sarvam", "mayura", [], "Sarvam API key missing"),
    ("groq", "mayura", [], "Sarvam API key missing"),   # mayura alone needs Sarvam
    ("groq", "gemini", ["GROQ_API_KEY"], "Gemini API key missing"),
    ("groq", "gemini", ["GEMINI_API_KEY"], "Groq API key missing"),
])
def test_a_missing_credential_is_a_configerror_before_a_run_can_start(
    stt, mt, configured, message, tmp_path, monkeypatch
):
    """The regression this closes: providers read their key lazily, so with no key
    the ConfigError escaped pipeline/live.py's worker thread, which catches only
    QuotaError and AuthError. The chunk's result never landed, the coordinator
    waited on it forever, and no end event was ever emitted — a run unreachable
    except by force-quitting the app. Requiring the credential here makes it a 400
    raised before any thread exists."""
    _empty_env(tmp_path, monkeypatch)
    for key in ("SARVAM_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key in configured:
        monkeypatch.setenv(key, "configured")

    with pytest.raises(ConfigError, match=message):
        build_run({"mode": "live", "stt": stt, "mt": mt, "langs": [],
                   "target_s": 8.0, "max_chunk_s": 28.0, "min_chunk_s": 5.0})


def test_a_free_tier_run_demands_no_sarvam_key(tmp_path, monkeypatch):
    # The reason the check asks per backend rather than for one key: picking the
    # free tiers must not demand an account the user does not have.
    _empty_env(tmp_path, monkeypatch)
    for key in ("SARVAM_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("GEMINI_API_KEY", "m")

    settings, *_ = build_run({"mode": "recording", "stt": "groq", "mt": "gemini",
                              "langs": [], "max_chunk_s": 28.0, "min_chunk_s": 5.0})
    assert settings.stt_provider == "groq"
    assert settings.api_key is None


def test_require_keys_is_the_shared_dispatch_not_a_second_copy():
    """config.require_keys is the one place that knows which key a backend needs;
    cli.py calls the same function, so adding a provider cannot leave the UI
    demanding a key it should not."""
    from omnilingual.config import require_keys

    settings = load_settings(env={})
    with pytest.raises(ConfigError, match="Sarvam API key missing"):
        require_keys(settings)
    ready = load_settings(env={"SARVAM_API_KEY": "k"})
    require_keys(ready)  # returns; the panel's 200 stands


# --- the output path a second live run would collide with ------------------


@respx.mock
def test_a_second_live_run_to_the_same_path_renames_instead_of_failing(respx_mock, tmp_path):
    """render/live.py opens with O_EXCL, so aiming a second run at an existing
    path dies with FileExistsError *after* the start request answered 200. The
    ordinary workflow — the same output today and again tomorrow — works in the
    CLI and must work here."""
    _route(respx_mock)
    config, stt, mt = _providers()
    out = tmp_path / "standup.md"
    paths = []
    for index in range(2):
        pcm = _pcm_3x20()
        factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm),
                                              gated=_gaps())
        runner, queue, opts, factory = _live(tmp_path, factory)
        opts = replace(opts, out=out, work_root=tmp_path / f"work{index}")
        runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                          opts=opts, capture_factory=factory)
        runner.join(timeout=120)
        assert runner.status == "done", [m for m in queue.of("log")]
        paths.append(runner.out_path)

    assert paths[0] == out
    assert paths[1] != out, "the second run must not target the first one's file"
    assert paths[1].is_file() and out.is_file(), "both transcripts survive"
    assert "· growing" not in paths[1].read_text(encoding="utf-8")
    # The start request answers with the path actually written, so the page links
    # to the new file rather than the one that was left alone.
    assert queue.of("end")[0]["out"] == str(paths[1])
    # The warning must name BOTH: the file that was in the way, and the one being
    # written. Naming the new name twice ("standup-2132.md exists; writing
    # standup-2132.md instead") is nonsense and hides the fact that the filename
    # the user chose was taken.
    warning = next(m["message"] for m in queue.of("log")
                   if "instead" in m["message"])
    assert warning == (f"{out.name} already exists; writing {paths[1].name} "
                       f"instead")
    assert out.name != paths[1].name
    assert "· growing" not in paths[1].read_text(encoding="utf-8")


def test_free_output_path_only_renames_on_a_collision(tmp_path):
    absent = tmp_path / "standup.md"
    assert free_output_path(absent) == absent

    occupied = tmp_path / "standup.md"
    occupied.write_text("previous transcript", encoding="utf-8")
    renamed = free_output_path(occupied)
    assert renamed != occupied
    assert renamed.suffix == ".md" and renamed.stem.startswith("standup-")
    # The candidate is offered, not created: the previous transcript stays exactly
    # where it was, and nothing is rewritten or moved until a run claims the name.
    assert not renamed.exists()
    assert occupied.read_text(encoding="utf-8") == "previous transcript"

    renamed.write_text("x", encoding="utf-8")
    again = free_output_path(occupied)
    assert again != renamed, "a second collision must not reuse the first name"


# --- the thread-to-event-loop bridge ---------------------------------------


@respx.mock
def test_events_bridge_to_a_real_loop_in_order(respx_mock, tmp_path):
    """The handoff this module exists for: pipeline threads -> call_soon_threadsafe
    -> an asyncio queue the server drains. FIFO order is the contract, because the
    page appends rows as they arrive and a reordering would show a shuffled
    transcript."""
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    config, stt, mt = _providers()

    loop = asyncio.new_event_loop()
    queue = asyncio.Queue()
    pump = threading.Thread(target=loop.run_forever, daemon=True)
    pump.start()
    try:
        runner = SessionRunner(run_id="r-loop", queue=queue, loop=loop)
        opts = LiveOptions(out=tmp_path / "meeting.md", stt_workers=1)
        runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                          opts=opts, capture_factory=factory)
        runner.join(timeout=120)
        # Let every already-queued callback run before reading the queue.
        deadline = time.monotonic() + 10
        while not queue.empty() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)
        delivered = []
        while not queue.empty():
            delivered.append(queue.get_nowait())
    finally:
        loop.call_soon_threadsafe(loop.stop)
        pump.join(timeout=10)
        loop.close()

    assert runner.status == "done"
    seqs = [m["seq"] for m in delivered if m["type"] == "segment"]
    assert seqs == sorted(seqs) and len(seqs) >= 3
    texts = [m["text"] for m in delivered if m["type"] == "segment"]
    assert texts.index("Bravo") < texts.index("Vanakkam")
    assert delivered[0]["type"] == "state"
    assert delivered[-1]["type"] == "end", "end must survive the bridge too"


@respx.mock
def test_a_closed_loop_costs_the_updates_not_the_run(respx_mock, tmp_path):
    """A page that disconnects, or an app quitting, closes the loop underneath a
    live run. Delivery then raises RuntimeError on every event; the meeting must
    still finish and still leave its transcript."""
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    config, stt, mt = _providers()

    loop = asyncio.new_event_loop()  # never run: start_live's own state lands
    runner = SessionRunner(run_id="r-closed", queue=asyncio.Queue(), loop=loop)
    opts = LiveOptions(out=tmp_path / "meeting.md", stt_workers=1)
    runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                      opts=opts, capture_factory=factory)
    loop.close()  # mid-run, with capture still feeding chunks
    runner.join(timeout=120)

    assert runner.status == "done"
    assert "Bravo" in (tmp_path / "meeting.md").read_text(encoding="utf-8")


# --- stopping: only the mode that can stop says so ------------------------


@respx.mock
def test_stop_does_not_claim_to_stop_a_batch_run(respx_mock, tmp_path):
    """Only run_live was given a stop channel. Announcing "stopping" for a batch run
    would leave the page waiting for an interruption that never comes."""
    _route(respx_mock)
    recording = tmp_path / "meeting.wav"
    make_wav(recording, [("tone", 3.0)])
    config, stt, mt = _providers()
    runner = SessionRunner(run_id="r-stop", queue=Collector())

    runner.start_recording(settings=config, stt=stt, translator=mt, diarizer=None,
                           source=recording, work_root=tmp_path / "work",
                           out=tmp_path / "batch.md")
    runner.stop()
    assert runner.status != "stopping"
    runner.join(timeout=300)

    assert runner.status == "done"
    assert (tmp_path / "batch.md").is_file()


@respx.mock
def test_stop_before_start_does_not_poison_the_next_run(respx_mock, tmp_path):
    """A stop that arrived while the runner was idle belongs to nothing. Carrying
    it into a run makes that run stop on its first read.

    This has to be driven through start_live: the batch loop takes no stop channel,
    so a recording run cannot observe the flag at all and the test would pass with
    the fix deleted. On a live run the difference is a run that reports "done" and
    writes a transcript with nothing in it.
    """
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    config, stt, mt = _providers()
    runner = SessionRunner(run_id="r-poison", queue=Collector())
    runner.stop()  # stray stop, while the runner is still idle

    opts = LiveOptions(out=tmp_path / "meeting.md", stt_workers=1)
    runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                      opts=opts, capture_factory=factory)
    runner.join(timeout=120)

    assert runner.status == "done"
    segments = runner._queue.of("segment")
    assert len(segments) >= 1, (
        "a poisoned stop flag makes a live run report 'done' with zero segments, "
        "which is a meeting that recorded nothing")
    assert "Bravo" in (tmp_path / "meeting.md").read_text(encoding="utf-8")


# --- the english-only sibling, on a real path -----------------------------


@respx.mock
def test_recording_with_english_only_writes_the_sibling_through_the_renderer(
    respx_mock, tmp_path
):
    """render_english_only had no coverage on any path, and it is the file the UI
    promises to produce for --english-only."""
    from omnilingual.render.markdown import render_english_only

    _route(respx_mock)
    recording = tmp_path / "meeting.wav"
    make_wav(recording, [("tone", 3.0)])
    config, stt, mt = _providers()
    queue = Collector()
    runner = SessionRunner(run_id="r-en", queue=queue)
    out = tmp_path / "ui.md"

    runner.start_recording(settings=config, stt=stt, translator=mt, diarizer=None,
                           source=recording, work_root=tmp_path / "work", out=out,
                           english_only=True)
    runner.join(timeout=300)

    sibling = out.with_suffix(".en.md")
    assert sibling.is_file()
    body = sibling.read_text(encoding="utf-8")
    assert body.startswith("# Meeting transcript — meeting.wav (English)")
    assert "->en] Bravo" in body, "English only: the original text is not repeated"
    assert out.read_text(encoding="utf-8") != body
    # And it is the pipeline's renderer, not the UI's: the two agree on the same
    # Transcript because the CLI would produce the same bytes.
    assert render_english_only.__module__ == "omnilingual.render.markdown"


@respx.mock
def test_recording_without_english_only_writes_no_sibling(respx_mock, tmp_path):
    _route(respx_mock)
    recording = tmp_path / "meeting.wav"
    make_wav(recording, [("tone", 3.0)])
    config, stt, mt = _providers()
    out = tmp_path / "ui.md"
    runner = SessionRunner(run_id="r-noen", queue=Collector())
    runner.start_recording(settings=config, stt=stt, translator=mt, diarizer=None,
                           source=recording, work_root=tmp_path / "work", out=out)
    runner.join(timeout=300)

    assert out.is_file()
    assert not out.with_suffix(".en.md").exists()


# --- a halt is the marker AND the status ----------------------------------


def test_a_halt_needs_both_the_marker_and_the_failed_status():
    # run_live writes the marker into a stt_failed segment and nothing else
    # produces that text, so the conjunction cannot misfire. Matching the text
    # alone would call a halt any future provider error carrying the string, and
    # matching the status alone would call every failed chunk a halt — the exact
    # confusion this state exists to avoid.
    from omnilingual.pipeline.live import _HALT_TEXT
    from omnilingual.ui.session import _is_halt

    chunk = Chunk(idx=0, start_s=0.0, end_s=4.0, wav_path="x.wav")
    marker = _HALT_TEXT["cost"]
    assert _is_halt(Segment(chunk, "unknown", 0.0, marker, None, "stt_failed"))
    assert not _is_halt(Segment(chunk, "unknown", 0.0, marker, None, "ok")), (
        "a marker string with a healthy status is not a halt")
    assert not _is_halt(Segment(chunk, "unknown", 0.0, "[transcription failed]",
                                None, "stt_failed")), (
        "an ordinary failed chunk is not a halt")
    assert not _is_halt(Segment(chunk, "hi-IN", 0.9, "Bravo", "Hello", "ok"))


# --- the failed/done distinction is about whether a file exists ------------


@respx.mock
def test_a_capture_that_never_opens_is_failed_because_no_file_exists(respx_mock, tmp_path):
    """run_live returns 1 for exactly one reason: the capture never opened, so it
    never created the writer and there is no transcript. That is what separates it
    from exit 2, which means a valid file with segments that need attention —
    calling that 'failed' would make the UI offer to recover something it must not."""
    from omnilingual.audio.live_capture import CaptureError

    class DeadCapture(FakeCapture):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, blocks=[], gated=[], **kwargs)

        def open(self):
            raise CaptureError("no such device")

    config, stt, mt = _providers()
    queue = Collector()
    runner = SessionRunner(run_id="r-noopen", queue=queue)
    out = tmp_path / "meeting.md"
    opts = LiveOptions(out=out, stt_workers=1)

    runner.start_live(settings=config, stt=stt, translator=mt, diarizer=None,
                      opts=opts,
                      capture_factory=lambda *a, **k: DeadCapture(*a, **k))
    runner.join(timeout=60)

    assert runner.status == "failed"
    assert queue.of("end")[0]["exit_code"] == 1
    assert not out.exists(), "no writer, so no file — that is why this is 'failed'"
    assert any("capture failed" in m["message"] for m in queue.of("status"))
    # The directory exists and holds a session.json, but nothing is sealed in it,
    # so recovery must not be offered for it.
    assert runner.session_dir is not None
    assert runner.recoverable is False
    assert queue.of("end")[0]["recoverable"] is False


def test_recover_reports_a_directory_that_is_not_a_live_session(tmp_path):
    """The Recover button is fed whatever the runs list hands it, so a directory
    that is not a session has to come back as a reported failure rather than an
    exception out of the request handler."""
    config, stt, mt = _providers()
    queue = Collector()
    runner = SessionRunner(run_id="r-bogus", queue=queue)
    bogus = tmp_path / "not-a-session"
    bogus.mkdir()

    runner.recover(session_dir=bogus, settings=config, stt=stt, translator=mt,
                   diarizer=None, out=tmp_path / "recovered.md")
    runner.join(timeout=60)

    assert runner.status == "failed"
    assert queue.of("end")[0]["exit_code"] != 0
    assert any("live session" in m["message"] for m in queue.of("log"))
    assert not (tmp_path / "recovered.md").exists()
    assert queue.of("state")[-1]["error"]


# --- every validation failure is one exception type -----------------------


@pytest.mark.parametrize("override,field", [
    ({"max_chunk_s": "abc"}, "max-chunk-s"),
    ({"min_chunk_s": "abc"}, "min-chunk-s"),
    ({"target_s": "eight"}, "target-s"),
    ({"num_speakers": "three"}, "num-speakers"),
    ({"num_speakers": [3]}, "num-speakers"),
    ({"max_chunk_s": {"v": 28}}, "max-chunk-s"),
])
def test_a_numeric_field_that_will_not_coerce_is_a_configerror(override, field,
                                                               tmp_path, monkeypatch):
    """float("abc") is a ValueError and float({"v": 28}) is a TypeError, and the
    spec's 400 path can only catch ConfigError. One except clause has to be enough,
    so every way a payload can be malformed lands on the same type — and names the
    field, because the user is looking at a form.

    null and blank are not here: they mean "not provided" and fall back to the
    field's default, which the test below covers."""
    _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "k")
    panel = {"mode": "live", "stt": "sarvam", "mt": "mayura", "langs": [],
             "target_s": 8.0, "max_chunk_s": 28.0, "min_chunk_s": 5.0}
    panel.update(override)

    with pytest.raises(ConfigError, match=field):
        build_run(panel)


@pytest.mark.parametrize("override,field", [
    ({"target_s": "eight"}, "target-s"),
    ({"max_chunk_s": "abc"}, "max-chunk-s"),
    ({"noise_db": "loud"}, "noise-db"),
    ({"stt_workers": "many"}, "stt-workers"),
    ({"max_cost": "lots"}, "max-cost"),
])
def test_live_options_rejects_a_bad_number_the_same_way(override, field, tmp_path):
    """live_options reads a superset of build_run's fields, so it needs the same
    wrapper: stt_workers, noise_db and max_cost never reach build_run at all, and a
    ConfigError from live_options must still be the server's single 400."""
    with pytest.raises(ConfigError, match=field):
        live_options(override, out=tmp_path / "x.md")


@pytest.mark.parametrize("override,field", [
    ({"num_speakers": 1e400}, "num-speakers"),
    ({"num_speakers": float("inf")}, "num-speakers"),
])
def test_a_non_finite_speaker_count_is_a_configerror(override, field,
                                                     tmp_path, monkeypatch):
    """1e400 arrives as the float inf. int(inf) raises OverflowError, which is
    neither TypeError nor ValueError, so the old except tuple let it escape as a
    500 instead of the 400 the panel renders."""
    _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "k")
    panel = {"mode": "live", "stt": "sarvam", "mt": "mayura", "langs": []}
    panel.update(override)

    with pytest.raises(ConfigError, match=field):
        build_run(panel)


@pytest.mark.parametrize("bad", [1e400, float("inf"), float("nan"), "inf"])
def test_a_non_finite_cost_cap_is_a_configerror(bad, tmp_path):
    """float("1e400") *succeeds* and yields inf, so no except clause can catch it.
    Allowed through, it reaches snapshot()'s cost_cap, and Starlette's send_json
    does not pass allow_nan=False — so the wire carries a bare Infinity token,
    JSON.parse throws, and app.js's unguarded onmessage loses every later frame
    including `end`, while the run keeps spending quota."""
    with pytest.raises(ConfigError, match="max-cost"):
        live_options({"max_cost": bad}, out=tmp_path / "x.md")


def test_a_huge_integer_for_num_speakers_does_not_overflow(tmp_path, monkeypatch):
    """A >=309-digit integer for num_speakers (as_int=True) must not escape as
    a 500 via OverflowError. int(10**400) succeeds as a Python bigint, and with
    the fix math.isfinite is only called on floats, so the value is accepted
    mathematically finite — the OverflowError that previously escaped is gone."""
    _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "k")
    panel = {"mode": "live", "stt": "sarvam", "mt": "mayura", "langs": []}
    panel["num_speakers"] = int("1" + "0" * 400)
    # The fix: OverflowError must not escape; the value is accepted as a valid int.
    # Before the fix this raised OverflowError outside the try block.
    settings, *_ = build_run(panel)
    assert settings.num_speakers == panel["num_speakers"]


def test_as_int_with_inf_still_raises_configerror(tmp_path, monkeypatch):
    """1e400 as a float yields inf; int(inf) raises OverflowError, which is caught
    by the except clause — ConfigError must still be raised, not a 500."""
    _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "k")
    panel = {"mode": "live", "stt": "sarvam", "mt": "mayura", "langs": []}
    panel["num_speakers"] = 1e400  # float('inf')
    with pytest.raises(ConfigError, match="num-speakers"):
        build_run(panel)


def test_max_cost_inf_still_raises_configerror(tmp_path):
    """float('inf') for max_cost must still be rejected by the isfinite check."""
    with pytest.raises(ConfigError, match="max-cost"):
        live_options({"max_cost": float("inf")}, out=tmp_path / "x.md")


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_a_cleared_numeric_field_falls_back_to_its_default(blank, tmp_path, monkeypatch):
    """Absent, null and blank all mean "not provided". Rejecting them would fail
    the request for a form the user merely cleared, and propagating None would hand
    run_live a None where it does max(1, opts.stt_workers)."""
    _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "k")
    panel = {"mode": "live", "stt": "sarvam", "mt": "mayura", "langs": [],
             "max_chunk_s": blank, "min_chunk_s": 5.0, "target_s": 8.0}

    settings, *_ = build_run(panel)
    assert settings.max_chunk_s == 28.0
    opts = live_options({"stt_workers": blank, "noise_db": blank}, out=tmp_path / "y.md")
    assert opts.stt_workers == 2, "the CLI's default, not None and not 1"
    assert opts.noise_db == -35.0


def test_num_speakers_absent_still_means_auto_count(tmp_path, monkeypatch):
    # The one field whose absence is meaningful rather than a fallback: leaving the
    # speaker count out has always meant "let the diarizer decide", and only
    # load_settings decides whether a count is legal.
    _empty_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "k")

    settings, *_ = build_run({"mode": "recording", "stt": "sarvam",
                              "mt": "mayura", "langs": []})
    assert settings.num_speakers is None
    with pytest.raises(ConfigError):
        build_run({"mode": "recording", "stt": "sarvam", "mt": "mayura",
                   "langs": [], "num_speakers": 1})
