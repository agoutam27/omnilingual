import io
import sys
import threading
from pathlib import Path

import pytest

from omnilingual.config import ConfigError, load_settings
from omnilingual.diarize import sherpa
from omnilingual.diarize.base import Turn
from omnilingual.diarize.sherpa import SherpaDiarizer


@pytest.fixture
def settings():
    return load_settings(api_key="k", env={}, diarizer="sherpa")


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    p = tmp_path / "c.wav"
    p.write_bytes(b"RIFF....WAVEfake")
    return p


def test_provider_attrs_and_model_namespacing(settings):
    dz = SherpaDiarizer(settings, engine=lambda p: [Turn(0.0, 1.0, "0")])
    assert dz.model == "sherpa:pyannote-segmentation-3-0-3dspeaker-eres2net"


def test_diarize_maps_engine_turns(settings, wav):
    dz = SherpaDiarizer(
        settings,
        engine=lambda p: [Turn(0.0, 2.5, "1"), Turn(2.5, 5.0, "0")],
    )
    assert dz.diarize(wav) == [Turn(0.0, 2.5, "1"), Turn(2.5, 5.0, "0")]


def test_missing_extra_raises_config_error(settings, monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ConfigError, match="diarize"):
        SherpaDiarizer(settings)


def test_calls_are_serialized(settings, wav):
    entered = threading.Event()
    release = threading.Event()
    done_second = threading.Event()

    def engine(p):
        entered.set()
        release.wait(2)
        return [Turn(0.0, 1.0, "0")]

    dz = SherpaDiarizer(settings, engine=engine)
    t1 = threading.Thread(target=dz.diarize, args=(wav,))
    t1.start()
    assert entered.wait(2)

    def second():
        dz.diarize(wav)
        done_second.set()

    t2 = threading.Thread(target=second)
    t2.start()
    assert not done_second.wait(0.2)
    release.set()
    t1.join()
    t2.join()
    assert done_second.is_set()


def test_ensure_models_downloads_only_when_absent(settings, tmp_path, monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    calls: list[Path] = []

    def downloader(url: str, dest: Path) -> None:
        calls.append(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake-model")

    dz = SherpaDiarizer(
        settings, downloader=downloader, model_dir=tmp_path / "models"
    )
    dz.ensure_models()
    assert len(calls) == 2  # segmentation + embedding
    dz.ensure_models()
    assert len(calls) == 2  # cached: no second download


@pytest.mark.diarize
def test_real_engine_smoke(settings, tmp_path):
    """Runs the actual model; skipped unless the diarize extra is installed."""
    pytest.importorskip("sherpa_onnx")
    from tests.conftest import make_wav

    wav = tmp_path / "two.wav"
    make_wav(wav, [("tone", 5.0), ("silence", 1.0), ("tone", 5.0)])
    dz = SherpaDiarizer(settings)
    turns = dz.diarize(wav)
    assert isinstance(turns, list)
    assert {t.speaker for t in turns} <= {"0", "1", "2", "3"}


# --- download progress --------------------------------------------------------


class _Clock:
    """Drives the throttle so progress cadence is asserted, not slept through."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _progress(monkeypatch, clock, total, *, tty):
    err = _Tty() if tty else io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sherpa.time, "monotonic", clock)
    return sherpa._DownloadProgress("seg.onnx", total), err


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(sherpa.time, "monotonic", c)
    return c


def test_progress_announces_size_before_the_first_chunk(monkeypatch, clock):
    """The whole point: a silent ~46 MB fetch must never look like a hang."""
    prog, err = _progress(monkeypatch, clock, 7_000_000, tty=False)
    prog.start()
    assert err.getvalue() == "downloading seg.onnx (7.0 MB)\n"


def test_progress_reports_percent_size_and_rate_when_redirected(monkeypatch, clock):
    prog, err = _progress(monkeypatch, clock, 4_000_000, tty=False)
    prog.start()
    prog.advance(2_000_000)
    clock.now = 2.0
    prog.advance(2_000_000)
    lines = err.getvalue().splitlines()
    assert lines[-1] == "  seg.onnx 100% 4.0 MB/4.0 MB at 2.00 MB/s"
    assert all("\r" not in line for line in lines)


def test_progress_rewrites_in_place_on_a_tty(monkeypatch, clock):
    prog, err = _progress(monkeypatch, clock, 4_000_000, tty=True)
    prog.start()
    clock.now = 1.0
    prog.advance(1_000_000)
    out = err.getvalue()
    assert out.startswith("downloading seg.onnx (4.0 MB)\n")
    assert "\r  seg.onnx 25% 1.0 MB/4.0 MB" in out
    prog.close()
    assert err.getvalue().endswith("\n")


def test_progress_throttles_chunks_within_the_interval(monkeypatch, clock):
    prog, err = _progress(monkeypatch, clock, 100_000_000, tty=False)
    prog.start()
    for _ in range(20):
        clock.now += 0.01
        prog.advance(1_000_000)
    assert err.getvalue().splitlines()[1:] == []
    clock.now += 1.0
    prog.advance(1_000_000)
    assert len(err.getvalue().splitlines()[1:]) == 1


def test_progress_without_content_length_omits_percent(monkeypatch, clock):
    prog, err = _progress(monkeypatch, clock, None, tty=False)
    prog.start()
    assert "size" not in err.getvalue()
    clock.now = 1.0
    prog.advance(3_000_000)
    assert err.getvalue().splitlines()[-1] == "  seg.onnx 3.0 MB at 3.00 MB/s"
