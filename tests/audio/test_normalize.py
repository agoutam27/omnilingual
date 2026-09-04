import wave
from pathlib import Path

import pytest

from omnilingual.audio.normalize import FfmpegError, FfmpegMissingError, ensure_ffmpeg, normalize, probe_duration
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


@requires_ffmpeg
def test_normalize_raises_on_bad_audio(tmp_path: Path):
    bad = tmp_path / "bad.m4a"
    bad.write_bytes(b"not audio")
    dst = tmp_path / "out.wav"
    with pytest.raises(FfmpegError) as ei:
        normalize(bad, dst)
    assert "ffmpeg failed" in str(ei.value)


@requires_ffmpeg
def test_normalize_leaves_no_part_file(tmp_path: Path):
    src = make_wav(tmp_path / "in.wav", [("tone", 1.0)], rate=44100, channels=2)
    dst = tmp_path / "out" / "normalized.wav"
    normalize(src, dst)
    assert dst.exists()
    assert list(dst.parent.glob("*.part")) == []


@requires_ffmpeg
def test_normalize_cleans_up_part_file_on_failure(tmp_path: Path):
    bad = tmp_path / "bad.m4a"
    bad.write_bytes(b"not audio")
    dst = tmp_path / "out" / "normalized.wav"
    with pytest.raises(FfmpegError):
        normalize(bad, dst)
    assert not dst.exists()
    assert list(dst.parent.glob("*.part")) == []


def test_normalize_never_leaves_a_truncated_destination(tmp_path: Path, monkeypatch):
    """ffmpeg dying mid-write must not leave a half-WAV that later runs would trust."""
    dst = tmp_path / "out" / "normalized.wav"

    def half_written_then_fail(cmd: list[str]):
        Path(cmd[-1]).write_bytes(b"RIFF-truncated")  # ffmpeg's partial output
        raise FfmpegError("ffmpeg failed (exit 255): killed")

    monkeypatch.setattr("omnilingual.audio.normalize.run_tool", half_written_then_fail)
    with pytest.raises(FfmpegError):
        normalize(tmp_path / "in.m4a", dst)
    assert not dst.exists()
    assert list(dst.parent.glob("*")) == []
