import pytest

from omnilingual.config import ConfigError, load_settings
from omnilingual.stt import build_stt
from omnilingual.stt.groq import GroqSTT
from omnilingual.stt.sarvam import SarvamSTT


def test_default_provider_is_sarvam_with_default_model():
    stt = build_stt(load_settings(api_key="k", env={}))
    assert isinstance(stt, SarvamSTT)
    assert stt.model == "saaras:v4"


def test_groq_provider_selected():
    stt = build_stt(load_settings(env={"GROQ_API_KEY": "g"}, stt_provider="groq"))
    assert isinstance(stt, GroqSTT)


def test_mlx_provider_constructs_when_extra_present(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    from omnilingual.stt.mlx_whisper import MlxWhisperSTT

    stt = build_stt(load_settings(api_key="k", env={}, stt_provider="mlx-whisper"))
    assert isinstance(stt, MlxWhisperSTT)
    assert stt.model == "mlx-whisper:mlx-community/whisper-large-v3-turbo"


def test_mlx_provider_without_extra_explains_install(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ConfigError, match="local-stt"):
        build_stt(load_settings(api_key="k", env={}, stt_provider="mlx-whisper"))


def test_faster_whisper_provider_constructs_when_extra_present(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    from omnilingual.stt.faster_whisper import FasterWhisperSTT

    stt = build_stt(load_settings(api_key="k", env={}, stt_provider="faster-whisper"))
    assert isinstance(stt, FasterWhisperSTT)
    assert stt.model == "faster-whisper:small"


def test_faster_whisper_provider_without_extra_explains_install(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ConfigError, match="local-stt"):
        build_stt(load_settings(api_key="k", env={}, stt_provider="faster-whisper"))


def test_unknown_provider_rejected():
    with pytest.raises(ConfigError, match="unknown STT provider"):
        load_settings(env={}, stt_provider="bogus")
