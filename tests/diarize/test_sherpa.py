import threading
from pathlib import Path

import pytest

from omnilingual.config import ConfigError, load_settings
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
