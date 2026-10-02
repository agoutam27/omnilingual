from pathlib import Path

import pytest
from typer.testing import CliRunner

from omnilingual import cli
from omnilingual import config
from omnilingual.models import Chunk, Cost, Segment, Transcript

runner = CliRunner()

DEFAULTS_IN = "\n" * 5  # stt, stt model, mt, mt model, diarize=no


def _transcript(source: Path) -> Transcript:
    seg = Segment(Chunk(0, 0, 10.0, Path("x")), "hi-IN", 0.9, "क", "EN", "ok")
    return Transcript(source=source, duration_s=10.0, segments=[seg], cost=Cost(0, 0, 0.5))


@pytest.fixture(autouse=True)
def free_tier_keys(monkeypatch):
    """Every test picks free-tier backends, so give them the keys those demand."""
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("GEMINI_API_KEY", "gk")
    monkeypatch.setenv("SARVAM_API_KEY", "sk")


@pytest.fixture
def rec(tmp_path: Path) -> Path:
    p = tmp_path / "meeting.m4a"
    p.write_bytes(b"fake")
    return p


@pytest.fixture
def seen(monkeypatch):
    """Stub ffmpeg, both builders and the run pipeline, recording what each receives."""
    seen = {"stt": [], "mt": [], "dz": [], "run": []}

    def note(key):
        def record(settings):
            seen[key].append(settings)
            return object()
        return record

    monkeypatch.setattr(cli, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(cli, "prepare",
                        lambda source, wd, s: (10.0, [Chunk(0, 0, 10.0, Path("a"))]))
    monkeypatch.setattr(cli, "build_stt", note("stt"))
    monkeypatch.setattr(cli, "build_translator", note("mt"))
    monkeypatch.setattr(cli, "build_diarizer", note("dz"))

    def fake_run(source, wd, settings, stt, translator, cache, progress=None, diarizer=None):
        seen["run"].append((settings, diarizer))
        return _transcript(source)
    monkeypatch.setattr(cli, "run", fake_run)
    return seen


@pytest.fixture
def live_seen(monkeypatch):
    seen = {}

    def fake_run_live(opts, settings, stt, translator, diarizer=None, status=None):
        seen["opts"] = opts
        seen["settings"] = settings
        seen["diarizer"] = diarizer
        return 0

    monkeypatch.setattr(cli, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(cli, "build_stt", lambda s: object())
    monkeypatch.setattr(cli, "build_translator", lambda s: object())
    monkeypatch.setattr(cli, "build_diarizer", lambda s: object())
    monkeypatch.setattr(cli, "run_live", fake_run_live)
    return seen


def _no_prompt(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a prompt ran without --ask")
    monkeypatch.setattr(cli, "_choose", boom)
    monkeypatch.setattr(cli.typer, "prompt", boom)


# --- transcribe --ask ------------------------------------------------------------


def test_ask_answers_reach_the_builders_and_settings(rec, seen, tmp_path):
    out = tmp_path / "asked.md"
    answers = ["groq", config.DEFAULT_STT_MODELS["groq"], "gemini",
               config.DEFAULT_MT_MODELS["gemini"], "yes", "3", str(out)]
    result = runner.invoke(cli.app, ["transcribe", str(rec), "--api-key", "k", "--ask"],
                           input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    # The builders are handed the chosen values, not the flag defaults.
    assert seen["stt"][0].stt_provider == "groq"
    assert seen["stt"][0].resolved_stt_model == config.DEFAULT_STT_MODELS["groq"]
    assert seen["mt"][0].mt_provider == "gemini"
    assert seen["mt"][0].resolved_mt_model == config.DEFAULT_MT_MODELS["gemini"]
    settings, diarizer = seen["run"][0]
    assert settings.diarizer == "sherpa"
    assert settings.num_speakers == 3
    assert diarizer is not None
    assert out.exists(), "the prompted output path is where the transcript lands"


def test_ask_accepts_a_bare_index(rec, seen):
    first_stt, first_mt = config.STT_PROVIDERS[0], config.MT_PROVIDERS[0]
    answers = ["1", "", "1", "", "no", str(rec.with_suffix(".md"))]
    result = runner.invoke(cli.app, ["transcribe", str(rec), "--api-key", "k", "--ask"],
                           input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    settings, _ = seen["run"][0]
    assert settings.stt_provider == first_stt
    assert settings.mt_provider == first_mt
    assert settings.diarizer is None


def test_ask_all_defaults_equal_the_non_interactive_run(rec, seen):
    guard = pytest.MonkeyPatch()
    try:
        _no_prompt(guard)
        plain = runner.invoke(cli.app, [str(rec), "--api-key", "k"])
        assert plain.exit_code == 0, plain.output
    finally:
        guard.undo()
    asked = runner.invoke(cli.app, ["transcribe", str(rec), "--api-key", "k", "--ask"],
                          input=DEFAULTS_IN + "\n")
    assert asked.exit_code == 0, asked.output
    plain_settings = seen["run"][0]
    asked_settings = seen["run"][1]
    assert asked_settings == plain_settings
    assert asked_settings[0].stt_provider == config.STT_PROVIDERS[0]
    assert asked_settings[0].mt_provider == config.MT_PROVIDERS[0]
    assert asked_settings[0].stt_model is None
    assert asked_settings[0].mt_model is None
    assert asked_settings[0].diarizer is None
    assert rec.with_suffix(".md").exists()


def test_ask_shows_the_current_default_and_keeps_flag_values(rec, seen, tmp_path):
    given = tmp_path / "given.md"
    result = runner.invoke(
        cli.app,
        ["transcribe", str(rec), "--api-key", "k", "--ask",
         "--stt", "faster-whisper", "--diarize", "--speakers", "5", "--out", str(given)],
        input="\n" * 7)
    assert result.exit_code == 0, result.output
    assert "choice [faster-whisper]:" in result.output, "the --stt value is the shown default"
    assert "choice [5]" in result.output, "the --speakers value is the default the user sees"
    settings, diarizer = seen["run"][0]
    assert settings.stt_provider == "faster-whisper"
    assert settings.num_speakers == 5
    assert settings.diarizer == "sherpa"
    assert given.exists()


def test_ask_overrides_an_explicit_flag(rec, seen):
    answers = ["groq", config.DEFAULT_STT_MODELS["groq"], "", "", "no",
               str(rec.with_suffix(".md"))]
    result = runner.invoke(
        cli.app,
        ["transcribe", str(rec), "--api-key", "k", "--ask",
         "--stt", "faster-whisper", "--mt", "mayura"],
        input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    assert seen["run"][0][0].stt_provider == "groq"
    assert seen["run"][0][0].mt_provider == config.MT_PROVIDERS[0]


def test_choice_lists_are_read_from_the_config_registries(rec, seen, monkeypatch):
    monkeypatch.setattr(cli.config, "MT_PROVIDERS", ("alpha", "beta"))
    monkeypatch.setattr(cli.config, "DEFAULT_MT_MODELS", {"alpha": "a1", "beta": "b1"})
    result = runner.invoke(
        cli.app,
        ["transcribe", str(rec), "--api-key", "k", "--ask", "--mt", "beta"],
        input="1\n\nalpha\n\nno\n" + str(rec.with_suffix(".md")) + "\n")
    assert result.exit_code == 0, result.output
    assert "alpha" in result.output and "beta" in result.output
    assert "a1" in result.output and "b1" in result.output
    assert "gemini" not in result.output and "mayura" not in result.output
    assert seen["run"][0][0].mt_provider == "alpha"


def test_ask_reprompts_on_unusable_input(rec, seen):
    answers = ["twelve", "99", "0", "1", "", "1", "", "no", str(rec.with_suffix(".md"))]
    result = runner.invoke(cli.app, ["transcribe", str(rec), "--api-key", "k", "--ask"],
                           input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    assert "enter a number 1-" in result.output
    assert seen["run"][0][0].stt_provider == config.STT_PROVIDERS[0]


def test_ask_with_exhausted_input_aborts_without_a_traceback(rec, seen):
    result = runner.invoke(cli.app, ["transcribe", str(rec), "--api-key", "k", "--ask"],
                           input="")
    assert result.exit_code == 1
    assert "Aborted." in result.output
    assert isinstance(result.exception, SystemExit)
    assert seen["run"] == [], "the pipeline is never reached"


def test_ask_missing_key_for_the_chosen_provider_still_fails_clearly(rec, seen, monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "sk")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    answers = ["groq", config.DEFAULT_STT_MODELS["groq"], "", "", "no",
               str(rec.with_suffix(".md"))]
    result = runner.invoke(cli.app, ["transcribe", str(rec), "--ask"],
                           input="\n".join(answers) + "\n")
    assert result.exit_code == 1
    assert "GROQ_API_KEY" in result.output
    assert seen["run"] == []


def test_ask_without_a_recording_is_a_usage_error(seen, monkeypatch):
    _no_prompt(monkeypatch)
    result = runner.invoke(cli.app, ["transcribe", "--ask"])
    assert result.exit_code == 1
    assert "RECORDING" in result.output


def test_without_ask_nothing_is_prompted_and_nothing_changes(rec, seen, monkeypatch):
    _no_prompt(monkeypatch)
    result = runner.invoke(cli.app, [str(rec), "--api-key", "k"])
    assert result.exit_code == 0, result.output
    settings, diarizer = seen["run"][0]
    assert settings.stt_provider == "sarvam"
    assert settings.mt_provider == "mayura"
    assert settings.stt_model is None and settings.mt_model is None
    assert settings.diarizer is None and settings.num_speakers is None
    assert diarizer is None


# --- live --ask ------------------------------------------------------------------


def test_live_ask_configures_the_run(live_seen, tmp_path):
    out = tmp_path / "asked-live.md"
    answers = ["groq", config.DEFAULT_STT_MODELS["groq"], "gemini",
               config.DEFAULT_MT_MODELS["gemini"], "yes", "2", str(out)]
    result = runner.invoke(cli.app, ["live", "--api-key", "k", "--ask"],
                           input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    settings = live_seen["settings"]
    assert settings.stt_provider == "groq"
    assert settings.mt_provider == "gemini"
    assert settings.diarizer == "sherpa"
    assert settings.num_speakers == 2
    assert live_seen["opts"].out == out
    assert live_seen["diarizer"] is not None


def test_live_ask_keeps_an_explicit_out(live_seen, tmp_path):
    given = tmp_path / "given.md"
    result = runner.invoke(cli.app, ["live", "--out", str(given), "--api-key", "k", "--ask"],
                           input=DEFAULTS_IN + "\n" + str(given) + "\n")
    assert result.exit_code == 0, result.output
    assert live_seen["opts"].out == given
    assert live_seen["settings"].stt_provider == "sarvam"
    assert live_seen["settings"].diarizer is None


def test_live_ask_defaults_out_when_the_flag_is_absent(live_seen, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["live", "--api-key", "k", "--ask"],
                           input=DEFAULTS_IN + "\n\n")
    assert result.exit_code == 0, result.output
    assert live_seen["opts"].out == Path("live.md")


def test_live_without_ask_still_requires_out(live_seen, monkeypatch):
    _no_prompt(monkeypatch)
    result = runner.invoke(cli.app, ["live", "--api-key", "k"])
    assert result.exit_code == 1
    assert "--out" in result.output


def test_live_without_ask_is_unchanged(live_seen, tmp_path, monkeypatch):
    _no_prompt(monkeypatch)
    given = tmp_path / "plain.md"
    result = runner.invoke(cli.app, ["live", "--out", str(given), "--api-key", "k",
                                     "--stt", "groq", "--mt", "gemini"])
    assert result.exit_code == 0, result.output
    assert live_seen["opts"].out == given
    assert live_seen["settings"].stt_provider == "groq"
    assert live_seen["settings"].mt_provider == "gemini"
    assert live_seen["diarizer"] is None
