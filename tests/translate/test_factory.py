import pytest

from omnilingual.config import ConfigError, load_settings
from omnilingual.translate import build_translator
from omnilingual.translate.gemini import GeminiTranslator
from omnilingual.translate.indictrans2 import IndicTrans2Translator
from omnilingual.translate.mayura import MayuraTranslator


def test_default_provider_is_mayura_with_default_model():
    tr = build_translator(load_settings(api_key="k", env={}))
    assert isinstance(tr, MayuraTranslator)
    assert tr.model == "mayura:v1"


def test_gemini_provider_selected():
    tr = build_translator(load_settings(env={"GEMINI_API_KEY": "gk"}, mt_provider="gemini"))
    assert isinstance(tr, GeminiTranslator)
    assert tr.model == "gemini:gemini-3.5-flash"


def test_gemini_model_override_is_namespaced():
    tr = build_translator(load_settings(env={}, mt_provider="gemini", mt_model="gemini-3.8-flash"))
    assert tr.model == "gemini:gemini-3.8-flash"


def test_mayura_model_override():
    tr = build_translator(load_settings(api_key="k", env={}, mt_model="sarvam-translate:v1"))
    assert tr.model == "sarvam-translate:v1"


def test_indictrans2_provider_selected(monkeypatch):
    import importlib.util

    # The constructor refuses to build without the local-mt extra, so stub the
    # probe: this test is about the factory, not about installing ctranslate2.
    local_mt = {"ctranslate2", "sentencepiece", "huggingface_hub"}
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name in local_mt else None,
    )
    tr = build_translator(load_settings(env={}, mt_provider="indictrans2"))
    assert isinstance(tr, IndicTrans2Translator)
    assert tr.model == "indictrans2:adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M"
    assert tr.inr_per_10k_chars == 0.0


def test_unknown_provider_rejected():
    with pytest.raises(ConfigError, match="unknown MT provider"):
        load_settings(env={}, mt_provider="bogus")