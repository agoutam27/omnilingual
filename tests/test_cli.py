from pathlib import Path

import pytest
from typer.testing import CliRunner

from omnilingual import cli
from omnilingual.http import QuotaError
from omnilingual.models import Chunk, Cost, Segment, Transcript

runner = CliRunner()


def _transcript(source: Path, statuses=("ok",)) -> Transcript:
    segs = [
        Segment(Chunk(i, i * 10.0, (i + 1) * 10.0, Path("x")), "hi-IN", 0.9, "क", "EN" if st == "ok" else None, st)
        for i, st in enumerate(statuses)
    ]
    return Transcript(source=source, duration_s=10.0 * len(segs), segments=segs, cost=Cost(0, 0, 0.5))


@pytest.fixture
def rec(tmp_path: Path) -> Path:
    p = tmp_path / "meeting.m4a"
    p.write_bytes(b"fake")
    return p


@pytest.fixture
def patched(monkeypatch, rec):
    """Stub out ffmpeg and network-touching pieces."""
    monkeypatch.setattr(cli, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(cli, "prepare", lambda source, wd, s: (20.0, [Chunk(0, 0, 10.0, Path("a")), Chunk(1, 10.0, 20.0, Path("b"))]))
    monkeypatch.setattr(cli, "SarvamSTT", lambda settings: object())
    monkeypatch.setattr(cli, "MayuraTranslator", lambda settings: object())
    calls = {}

    def fake_run(source, wd, settings, stt, translator, cache, progress=None):
        calls["settings"] = settings
        calls["wd"] = wd
        t = _transcript(source, calls.get("statuses", ("ok",)))
        if progress:
            for i, seg in enumerate(t.segments, 1):
                progress(i, len(t.segments), seg)
        return t

    monkeypatch.setattr(cli, "run", fake_run)
    return calls


def test_writes_markdown_and_exits_zero(rec, patched):
    result = runner.invoke(cli.app, [str(rec), "--api-key", "k"])
    assert result.exit_code == 0, result.output
    out = rec.with_suffix(".md")
    assert out.exists()
    assert "# Meeting transcript — meeting.m4a" in out.read_text(encoding="utf-8")
    assert "₹0.50" in result.output


def test_english_only_writes_second_file(rec, patched):
    result = runner.invoke(cli.app, [str(rec), "--api-key", "k", "--english-only"])
    assert result.exit_code == 0, result.output
    assert rec.with_suffix(".en.md").exists()


def test_custom_out_and_langs(rec, patched, tmp_path):
    out = tmp_path / "custom.md"
    result = runner.invoke(cli.app, [str(rec), "--api-key", "k", "--out", str(out), "--langs", "hi-IN,ta-IN"])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert patched["settings"].langs == ("hi-IN", "ta-IN")


def test_partial_success_exits_two(rec, patched):
    patched["statuses"] = ("ok", "stt_failed")
    result = runner.invoke(cli.app, [str(rec), "--api-key", "k"])
    assert result.exit_code == 2
    assert "1 segment(s) need attention" in result.output


def test_missing_key_exits_one(rec, patched, monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    result = runner.invoke(cli.app, [str(rec)])
    assert result.exit_code == 1
    assert "SARVAM_API_KEY" in result.output


def test_estimate_needs_no_key_and_makes_no_run(rec, patched, monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.setattr(cli, "run", lambda *a, **k: pytest.fail("run must not be called"))
    result = runner.invoke(cli.app, [str(rec), "--estimate"])
    assert result.exit_code == 0, result.output
    assert "2 chunks" in result.output
    assert "₹" in result.output


def test_quota_error_exits_one_with_resume_hint(rec, patched, monkeypatch):
    def boom(*a, **k):
        raise QuotaError("credits exhausted", 402)
    monkeypatch.setattr(cli, "run", boom)
    result = runner.invoke(cli.app, [str(rec), "--api-key", "k"])
    assert result.exit_code == 1
    assert "re-run" in result.output.lower()


def test_missing_ffmpeg_exits_one(rec, patched, monkeypatch):
    from omnilingual.audio.normalize import FfmpegMissingError
    def missing():
        raise FfmpegMissingError("ffmpeg not found on PATH. Install with: brew install ffmpeg")
    monkeypatch.setattr(cli, "ensure_ffmpeg", missing)
    result = runner.invoke(cli.app, [str(rec), "--api-key", "k"])
    assert result.exit_code == 1
    assert "brew install ffmpeg" in result.output


def test_ffmpeg_error_in_run_exits_one(rec, patched, monkeypatch):
    from omnilingual.audio.normalize import FfmpegError
    def boom(*a, **k):
        raise FfmpegError("ffmpeg failed (exit 1): Invalid data found")
    monkeypatch.setattr(cli, "run", boom)
    result = runner.invoke(cli.app, [str(rec), "--api-key", "k"])
    assert result.exit_code == 1
    assert "ffmpeg failed" in result.output


def test_ffmpeg_error_in_prepare_during_estimate_exits_one(rec, patched, monkeypatch):
    from omnilingual.audio.normalize import FfmpegError
    def boom(*a, **k):
        raise FfmpegError("ffmpeg failed (exit 1): Invalid data found")
    monkeypatch.setattr(cli, "prepare", boom)
    result = runner.invoke(cli.app, [str(rec), "--api-key", "k", "--estimate"])
    assert result.exit_code == 1
    assert "ffmpeg failed" in result.output


def test_missing_input_file_exits_one(tmp_path):
    result = runner.invoke(cli.app, [str(tmp_path / "nope.m4a"), "--api-key", "k"])
    assert result.exit_code != 0
