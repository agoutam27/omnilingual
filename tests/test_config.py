import pytest

from omnilingual.config import ConfigError, Settings, load_settings


def test_load_settings_reads_key_from_env():
    s = load_settings(env={"SARVAM_API_KEY": "k123"})
    assert s.api_key == "k123"
    assert s.stt_model == "saaras:v4"
    assert s.mt_model == "mayura:v1"
    assert s.max_chunk_s == 28.0
    assert s.min_chunk_s == 5.0


def test_explicit_key_beats_env():
    s = load_settings(api_key="explicit", env={"SARVAM_API_KEY": "fromenv"})
    assert s.api_key == "explicit"


def test_missing_key_is_allowed_until_required():
    s = load_settings(env={})
    assert s.api_key is None
    with pytest.raises(ConfigError):
        s.require_key()


def test_require_key_returns_key():
    assert load_settings(api_key="abc", env={}).require_key() == "abc"


def test_overrides_and_langs_tuple():
    s = load_settings(env={}, langs=["hi-IN", "ta-IN"], max_chunk_s=20.0)
    assert s.langs == ("hi-IN", "ta-IN")
    assert s.max_chunk_s == 20.0


def test_mt_char_limit_depends_on_model():
    assert load_settings(env={}).mt_char_limit == 1000
    assert load_settings(env={}, mt_model="sarvam-translate:v1").mt_char_limit == 2000


def test_settings_is_frozen():
    s = load_settings(env={})
    with pytest.raises(Exception):
        s.api_key = "x"  # type: ignore[misc]
