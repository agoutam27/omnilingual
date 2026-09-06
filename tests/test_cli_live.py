import json
import re

import pytest
from typer.testing import CliRunner

import omnilingual.cli as cli_mod
from omnilingual.cli import app
from omnilingual.models import Chunk, chunks_to_json
from tests.conftest import make_wav

runner = CliRunner()


class FakeSTT:
    model = "fake-stt"
    mode = "transcribe"

    def __init__(self, settings):
        self.calls = 0

    def transcribe(self, wav_path):
        from omnilingual.models import STTResult

        self.calls += 1
        return STTResult(lang="hi-IN", prob=0.9, text="Namaste")


class FakeTranslator:
    model = "fake-mt"

    def __init__(self, settings):
        pass

    def supports(self, lang):
        return True

    def to_english(self, text, src_lang):
        return f"[{src_lang}->en] {text}"


@pytest.fixture
def live_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_mod, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(cli_mod, "SarvamSTT", FakeSTT)
    monkeypatch.setattr(cli_mod, "MayuraTranslator", FakeTranslator)
    return tmp_path


def _session_dir(tmp_path, n=1):
    from omnilingual.pipeline import LIVE_SESSION_KIND

    session = tmp_path / "live-XYZ"
    (session / "live-chunks").mkdir(parents=True)
    chunks = []
    for i in range(n):
        wav = session / "live-chunks" / f"{i:04d}.wav"
        make_wav(wav, [("tone", 2.0)])
        chunks.append(Chunk(idx=i, start_s=float(2 * i),
                            end_s=float(2 * i + 2), wav_path=wav.resolve()))
    (session / "chunks.json").write_text(chunks_to_json(chunks), encoding="utf-8")
    (session / "session.json").write_text(json.dumps({
        "kind": LIVE_SESSION_KIND, "device": "Omnilingual"}), encoding="utf-8")
    return session


def test_live_validates_chunks_before_ffmpeg(live_cli, monkeypatch):
    from omnilingual.audio import normalize

    def boom():
        raise AssertionError("ffmpeg must not be touched on bad flags")

    monkeypatch.setattr(normalize, "ensure_ffmpeg", boom)
    monkeypatch.setattr(cli_mod, "ensure_ffmpeg", boom)
    res = runner.invoke(app, ["live", "--out", "m.md",
                              "--max-chunk-s", "30", "--min-chunk-s", "5"])
    assert res.exit_code == 1
    assert "--max-chunk-s must be < 30" in res.output


def test_live_validates_target_and_workers(live_cli):
    res = runner.invoke(app, ["live", "--out", "m.md", "--target-s", "60"])
    assert res.exit_code == 1
    assert "--target-s" in res.output
    res = runner.invoke(app, ["live", "--out", "m.md", "--stt-workers", "0"])
    assert res.exit_code == 1
    assert "--stt-workers" in res.output


def test_live_resolves_out_collision(live_cli, monkeypatch):
    seen = {}

    def fake_run_live(opts, *a, **k):
        seen["out"] = opts.out
        return 0

    monkeypatch.setattr(cli_mod, "run_live", fake_run_live)
    (live_cli / "m.md").write_text("precious", encoding="utf-8")
    res = runner.invoke(app, ["live", "--out", str(live_cli / "m.md"),
                              "--api-key", "k"])
    assert res.exit_code == 0
    assert re.fullmatch(r"m-\d{4}\.md", seen["out"].name)
    assert (live_cli / "m.md").read_text(encoding="utf-8") == "precious"


def test_live_passes_options_through(live_cli, monkeypatch):
    seen = {}

    def fake_run_live(opts, *a, **k):
        seen["opts"] = opts
        return 0

    monkeypatch.setattr(cli_mod, "run_live", fake_run_live)
    res = runner.invoke(app, ["live", "--out", str(live_cli / "m.md"),
                              "--input", "Mic", "--mic-only",
                              "--langs", "hi-IN,ta-IN", "--target-s", "10",
                              "--max-cost", "25", "--stt-workers", "3",
                              "--api-key", "k"])
    assert res.exit_code == 0
    o = seen["opts"]
    assert (o.device, o.mic_only, o.target_s, o.max_cost,
            o.stt_workers) == ("Mic", True, 10.0, 25.0, 3)
    # --langs rides settings (like the batch path), not LiveOptions.


def test_live_check_audio_probes_without_api(live_cli, monkeypatch):
    from omnilingual.audio.live_capture import BYTES_PER_SECOND

    def fail_settings(*a, **k):
        raise AssertionError("no API settings in --check-audio")

    monkeypatch.setattr(cli_mod, "load_settings", fail_settings)

    class FakeCap:
        def __init__(self, *a, **k):
            pass

        def open(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

        def read(self, n):
            assert n == BYTES_PER_SECOND
            return b"\x00" * n

        def close(self):
            pass

    monkeypatch.setattr(cli_mod, "LiveCapture", FakeCap)
    from omnilingual.audio.live_capture import AudioDevice

    monkeypatch.setattr(cli_mod, "parse_devices",
                        lambda text: [AudioDevice(index=2, name="Omnilingual")])
    import subprocess
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: type("R", (), {"stderr": "", "stdout": ""})())
    res = runner.invoke(app, ["live", "--out", "m.md", "--check-audio"])
    assert res.exit_code == 0
    assert "dBFS" in res.output


def test_live_check_audio_needs_no_out(live_cli, monkeypatch):
    from omnilingual.audio.live_capture import BYTES_PER_SECOND

    class FakeCap:
        def __init__(self, *a, **k):
            pass

        def open(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()
            return False

        def read(self, n):
            assert n == BYTES_PER_SECOND
            return b"\x00" * n

        def close(self):
            pass

    monkeypatch.setattr(cli_mod, "LiveCapture", FakeCap)
    from omnilingual.audio.live_capture import AudioDevice

    monkeypatch.setattr(cli_mod, "parse_devices",
                        lambda text: [AudioDevice(index=2, name="Omnilingual")])
    import subprocess
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: type("R", (), {"stderr": "", "stdout": ""})())
    res = runner.invoke(app, ["live", "--check-audio"])
    assert res.exit_code == 0, res.output
    assert "dBFS" in res.output


def test_live_requires_out_without_check_audio(live_cli):
    res = runner.invoke(app, ["live", "--api-key", "k"])
    assert res.exit_code == 1
    assert "--out" in res.output


def _check_audio_with_probe(live_cli, monkeypatch, probe: bytes):
    from omnilingual.audio.live_capture import BYTES_PER_SECOND

    assert len(probe) == BYTES_PER_SECOND

    class FakeCap:
        def __init__(self, *a, **k):
            pass

        def open(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

        def read(self, n):
            assert n == BYTES_PER_SECOND
            return probe

        def close(self):
            pass

    monkeypatch.setattr(cli_mod, "LiveCapture", FakeCap)
    from omnilingual.audio.live_capture import AudioDevice

    monkeypatch.setattr(cli_mod, "parse_devices",
                        lambda text: [AudioDevice(index=2, name="Omnilingual")])
    import subprocess
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: type("R", (), {"stderr": "", "stdout": ""})())
    return runner.invoke(app, ["live", "--check-audio"])


def test_live_check_audio_voice_verdict(live_cli, monkeypatch):
    from tests.conftest import raw_pcm

    res = _check_audio_with_probe(live_cli, monkeypatch, raw_pcm([("tone", 1.0)]))
    assert res.exit_code == 0, res.output
    assert "VOICE LIKELY" in res.output


def test_live_check_audio_hum_verdict(live_cli, monkeypatch):
    import math
    import struct

    n = 16000
    hum = struct.pack(f"<{n}h", *(int(8000 * math.sin(2 * math.pi * 100 * i / 16000))
                                  for i in range(n)))
    res = _check_audio_with_probe(live_cli, monkeypatch, hum)
    assert res.exit_code == 0, res.output
    assert "NO VOICE" in res.output


def test_transcribe_from_chunks_recovers_session(live_cli):
    session = _session_dir(live_cli, n=2)
    out = live_cli / "recovered.md"
    res = runner.invoke(app, ["transcribe", "--from-chunks", str(session),
                              "--out", str(out), "--api-key", "k"])
    assert res.exit_code == 0, res.output
    text = out.read_text(encoding="utf-8")
    # Each segment renders its text line plus the "> translation" quote line.
    assert text.count("> [hi-IN->en] Namaste") == 2
    assert "Namaste" in text


def test_transcribe_from_chunks_rejects_bad_dir(live_cli):
    res = runner.invoke(app, ["transcribe", "--from-chunks", str(live_cli)])
    assert res.exit_code == 1
    assert "live-session" in res.output
