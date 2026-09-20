import threading
from pathlib import Path

import pytest

from omnilingual.config import ConfigError, load_settings
from omnilingual.stt.mlx_whisper import MlxWhisperSTT


@pytest.fixture
def settings():
    return load_settings(api_key="k", env={}, stt_provider="mlx-whisper")


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    p = tmp_path / "c.wav"
    p.write_bytes(b"RIFF....WAVEfake")
    return p


def test_provider_attrs_and_default_model(settings):
    stt = MlxWhisperSTT(settings, decode=lambda p: ("x", "en", 1.0))
    assert stt.mode == "transcribe"
    assert stt.inr_per_hour == 0.0
    assert stt.model == "mlx-whisper:mlx-community/whisper-large-v3-turbo"


def test_model_override_is_namespaced():
    s = load_settings(env={}, stt_provider="mlx-whisper", stt_model="mlx-community/whisper-large-v3")
    assert MlxWhisperSTT(s, decode=lambda p: ("", "en", 1.0)).model == "mlx-whisper:mlx-community/whisper-large-v3"


def test_transcribe_maps_language_and_strips(settings, wav):
    stt = MlxWhisperSTT(settings, decode=lambda p: ("  नमस्ते  ", "hi", 0.87))
    r = stt.transcribe(wav)
    assert (r.text, r.lang, r.prob) == ("नमस्ते", "hi-IN", 0.87)


def test_missing_extra_raises_config_error(settings, monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ConfigError, match="local-stt"):
        MlxWhisperSTT(settings)


def test_calls_are_serialized(settings, wav):
    entered = threading.Event()
    release = threading.Event()
    done_second = threading.Event()

    def decode(p):
        entered.set()
        release.wait(2)
        return ("x", "en", 0.5)

    stt = MlxWhisperSTT(settings, decode=decode)
    t1 = threading.Thread(target=stt.transcribe, args=(wav,))
    t1.start()
    assert entered.wait(2)

    def second():
        stt.transcribe(wav)
        done_second.set()

    t2 = threading.Thread(target=second)
    t2.start()
    assert not done_second.wait(0.2)
    release.set()
    t1.join()
    t2.join()
    assert done_second.is_set()


@pytest.mark.mlx
def test_real_engine_smoke(settings, tmp_path):
    """Runs the actual model; skipped unless the local-stt extra is installed."""
    pytest.importorskip("mlx_whisper")
    from tests.conftest import make_wav

    wav = tmp_path / "tone.wav"
    make_wav(wav, [("tone", 1.0)])
    r = MlxWhisperSTT(settings).transcribe(wav)
    assert isinstance(r.lang, str) and r.lang
    assert 0.0 <= r.prob <= 1.0
