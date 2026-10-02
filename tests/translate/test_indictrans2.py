import pytest

from omnilingual.config import ConfigError, MT_CHAR_LIMITS, load_settings
from omnilingual.translate import build_translator
from omnilingual.translate.indictrans2 import SRC_TAGS, IndicTrans2Translator
from omnilingual.translate.mayura import SARVAM_TRANSLATE_LANGS


@pytest.fixture
def settings():
    return load_settings(env={}, mt_provider="indictrans2")


def _echo(pieces, src_tag):
    return [f"en<{src_tag}>{p}" for p in pieces]


def test_prices_as_free(settings):
    assert IndicTrans2Translator(settings, translate=_echo).inr_per_10k_chars == 0.0


def test_supports_indic_and_rejects_others(settings):
    tr = IndicTrans2Translator(settings, translate=_echo)
    assert tr.supports("hi-IN")
    assert tr.supports("ta-IN")
    assert tr.supports("od-IN")
    assert tr.supports("sat-IN")
    assert not tr.supports("en-IN")
    assert not tr.supports("hi")
    assert not tr.supports("unknown")


def test_covers_exactly_the_22_sarvam_indic_codes():
    assert set(SRC_TAGS) == SARVAM_TRANSLATE_LANGS - {"en-IN"}
    assert len(SRC_TAGS) == 22


def test_cache_key_is_namespaced_by_provider_model(settings):
    assert IndicTrans2Translator(settings, translate=_echo).model == (
        "indictrans2:adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M"
    )


def test_model_override_is_namespaced():
    s = load_settings(env={}, mt_provider="indictrans2", mt_model="some/other-ct2")
    assert IndicTrans2Translator(s, translate=_echo).model == "indictrans2:some/other-ct2"


def test_factory_builds_it(monkeypatch):
    import importlib.util

    local_mt = {"ctranslate2", "sentencepiece", "huggingface_hub"}
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name in local_mt else None,
    )
    tr = build_translator(load_settings(env={}, mt_provider="indictrans2", mt_model="adalat-ai/x"))
    assert tr.model == "indictrans2:adalat-ai/x"


def test_to_english_passes_the_language_tag_and_returns_english(settings):
    assert IndicTrans2Translator(settings, translate=_echo).to_english("नमस्ते", "hi-IN") == (
        "en<hin_Deva>नमस्ते"
    )


def test_splits_long_text_on_sentence_boundaries(settings):
    seen = []

    def spy(pieces, src_tag):
        seen.extend(pieces)
        return [f"t{i}" for i in range(len(pieces))]

    text = ("क" * 150 + "। ") * 3  # 456 chars in three sentences
    out = IndicTrans2Translator(settings, translate=spy).to_english(text, "hi-IN")
    limit = MT_CHAR_LIMITS["adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M"]
    assert len(seen) == 3
    assert all(len(p) <= limit for p in seen)
    assert out == "t0 t1 t2"


def test_char_limit_is_conservative_not_the_4000_default(settings):
    assert settings.mt_char_limit == 200
    seen = []
    IndicTrans2Translator(settings, translate=lambda p, t: seen.extend(p) or ["x"]).to_english(
        "क" * 2000, "hi-IN"
    )
    assert seen  # a single unpunctuated run is hard-split rather than dropped
    assert sum(len(p) for p in seen) == 2000


def test_blank_text_never_reaches_the_engine(settings):
    def boom(pieces, src_tag):
        raise AssertionError("engine called for blank text")

    assert IndicTrans2Translator(settings, translate=boom).to_english("   \n ", "hi-IN") == ""


def test_missing_extra_raises_config_error(settings, monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ConfigError, match="local-mt"):
        IndicTrans2Translator(settings)