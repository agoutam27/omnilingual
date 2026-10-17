import threading
from pathlib import Path

import pytest

from omnilingual.config import ConfigError, load_settings
from omnilingual.stt.indic_transcribe import IndicTranscribeSTT, _IndicEngine
from omnilingual.stt.langs import from_bcp47


@pytest.fixture
def settings():
    return load_settings(api_key="k", env={}, stt_provider="indic-transcribe")


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    p = tmp_path / "c.wav"
    p.write_bytes(b"RIFF....WAVEfake")
    return p


def test_provider_attrs_and_default_model(settings):
    stt = IndicTranscribeSTT(settings, decode=lambda p: ("x", "en", 1.0))
    assert stt.mode == "native"
    assert stt.inr_per_hour == 0.0
    assert stt.model == "indic-transcribe:bodhan-ai/indic-transcribe-core"


def test_model_override_is_namespaced():
    s = load_settings(env={}, stt_provider="indic-transcribe", stt_model="/models/itc")
    assert IndicTranscribeSTT(s, decode=lambda p: ("", "en", 1.0)).model == "indic-transcribe:/models/itc"


def test_transcribe_maps_language_and_strips(settings, wav):
    stt = IndicTranscribeSTT(settings, decode=lambda p: ("  नमस्ते  ", "hi", 0.87))
    r = stt.transcribe(wav)
    assert (r.text, r.lang, r.prob) == ("नमस्ते", "hi-IN", 0.87)


def test_odia_round_trips_through_the_iso_code():
    assert from_bcp47("od-IN") == "or"
    assert from_bcp47("en-IN") == "en"
    assert from_bcp47("xx-ZZ") == "xx-ZZ"


def test_missing_extra_names_what_is_missing(settings, monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ConfigError, match="indic-stt extra.*torch"):
        IndicTranscribeSTT(settings)


def test_calls_are_serialized(settings, wav):
    entered, release, done_second = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def decode(p):
        calls.append(1)
        entered.set()
        release.wait(2)
        return ("t", "hi", 1.0)

    stt = IndicTranscribeSTT(settings, decode=decode)
    t1 = threading.Thread(target=lambda: stt.transcribe(wav))
    t1.start()
    assert entered.wait(2)

    def second():
        stt.transcribe(wav)
        done_second.set()

    t2 = threading.Thread(target=second)
    t2.start()
    assert not done_second.wait(0.2)
    assert len(calls) == 1
    release.set()
    t1.join()
    t2.join()
    assert done_second.is_set()


def test_engine_restricts_language_id_to_known_langs(wav, monkeypatch):
    """--langs (BCP-47) reaches the model as its ISO codes; codes it never trained on
    are dropped, and an empty result means unrestricted rather than an empty filter."""
    import sys
    import types

    seen = {}

    class FakeASR:
        @classmethod
        def from_pretrained(cls, path, device, dtype):
            return cls()

        def transcribe(self, audio, lang, mode, return_lid, allowed_langs):
            seen.update(lang=lang, mode=mode, allowed=allowed_langs)
            return " hi ", {"lang": "ta", "topk": [("ta", 0.9), ("hi", 0.1)]}

    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        bfloat16="bf16", float32="fp32",
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    fake = types.ModuleType("indic_transcribe")
    fake.LANGUAGES = ("en", "hi", "ta", "or")
    fake.IndicTranscribe = FakeASR
    monkeypatch.setitem(sys.modules, "indic_transcribe", fake)
    monkeypatch.setattr(_IndicEngine, "_resolve", staticmethod(lambda m: str(wav.parent)))

    engine = _IndicEngine("m", ("hi-IN", "od-IN", "zz-IN"))
    assert engine(wav) == (" hi ", "ta", 0.9)
    assert seen == {"lang": None, "mode": "native", "allowed": ["hi", "or"]}

    assert _IndicEngine("m", ())._allowed is None
    assert _IndicEngine("m", ("zz-IN",))._allowed is None


def test_gated_repo_is_a_config_error(monkeypatch):
    import sys
    import types

    class GatedRepoError(Exception):
        pass

    def deny(model_id):
        raise GatedRepoError("403")

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = deny
    errors = types.ModuleType("huggingface_hub.errors")
    errors.GatedRepoError = GatedRepoError
    errors.RepositoryNotFoundError = type("RepositoryNotFoundError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    with pytest.raises(ConfigError, match="gated"):
        _IndicEngine._resolve("bodhan-ai/indic-transcribe-core")


@pytest.mark.indic_transcribe
def test_real_engine_smoke(settings, tmp_path):
    """Runs the actual model; needs the indic-stt extra and gated HF access."""
    pytest.importorskip("torch")
    from tests.conftest import make_wav

    wav = tmp_path / "tone.wav"
    make_wav(wav, [("tone", 2.0)])
    r = IndicTranscribeSTT(load_settings(env={}, stt_provider="indic-transcribe", langs=["hi-IN", "en-IN"])).transcribe(wav)
    assert r.lang in {"hi-IN", "en-IN"}
    assert 0.0 <= r.prob <= 1.0
