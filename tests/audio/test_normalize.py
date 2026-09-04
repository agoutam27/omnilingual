import wave
from pathlib import Path

import pytest

from omnilingual.audio.normalize import FfmpegMissingError, ensure_ffmpeg, normalize, probe_duration
from tests.conftest import make_wav, requires_ffmpeg


@requires_ffmpeg
def test_normalize_converts_to_16k_mono(tmp_path: Path):
    src = make_wav(tmp_path / "in.wav", [("tone", 2.0)], rate=44100, channels=2)
    dst = tmp_path / "out.wav"
    duration = normalize(src, dst)
    with wave.open(str(dst), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
    assert abs(duration - 2.0) < 0.1


@requires_ffmpeg
def test_probe_duration(tmp_path: Path):
    src = make_wav(tmp_path / "in.wav", [("tone", 1.0), ("silence", 0.5)])
    assert abs(probe_duration(src) - 1.5) < 0.05


@requires_ffmpeg
def test_ensure_ffmpeg_passes_when_installed():
    ensure_ffmpeg()


def test_ensure_ffmpeg_raises_when_missing(monkeypatch):
    monkeypatch.setattr("omnilingual.audio.normalize.shutil.which", lambda name: None)
    with pytest.raises(FfmpegMissingError) as ei:
        ensure_ffmpeg()
    assert "brew install ffmpeg" in str(ei.value)
