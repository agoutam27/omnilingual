import pytest

from omnilingual.config import (
    ConfigError,
    Settings,
    load_settings,
    validate_chunk_bounds,
    validate_target_s,
)


def test_load_settings_reads_key_from_env():
    s = load_settings(env={"SARVAM_API_KEY": "k123"})
    assert s.api_key == "k123"
    assert s.stt_model is None
    assert s.resolved_stt_model == "saaras:v4"
    assert s.stt_provider == "sarvam"
    assert s.mt_provider == "mayura"
    assert s.mt_model is None
    assert s.resolved_mt_model == "mayura:v1"
    assert s.max_chunk_s == 28.0
    assert s.min_chunk_s == 5.0


def test_groq_key_from_env_and_require():
    s = load_settings(env={"GROQ_API_KEY": "g"})
    assert s.groq_api_key == "g"
    assert s.require_groq_key() == "g"
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        load_settings(env={}).require_groq_key()


def test_resolved_stt_model_per_provider_and_override():
    assert load_settings(env={}, stt_provider="groq").resolved_stt_model == "whisper-large-v3-turbo"
    assert load_settings(env={}, stt_provider="mlx-whisper").resolved_stt_model == "mlx-community/whisper-large-v3-turbo"
    s = load_settings(env={}, stt_provider="groq", stt_model="whisper-large-v3")
    assert s.resolved_stt_model == "whisper-large-v3"


def test_unknown_stt_provider_rejected_at_load():
    with pytest.raises(ConfigError, match="unknown STT provider"):
        load_settings(env={}, stt_provider="bogus")


def test_diarizer_defaults_off():
    s = load_settings(env={})
    assert s.diarizer is None
    assert s.num_speakers is None


def test_num_speakers_needs_at_least_two():
    with pytest.raises(ConfigError, match="num_speakers"):
        load_settings(env={}, num_speakers=1)
    assert load_settings(env={}, num_speakers=3).num_speakers == 3


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
    assert load_settings(env={}, mt_provider="gemini").mt_char_limit == 4000


def test_resolved_mt_model_per_provider_and_override():
    assert load_settings(env={}, mt_provider="gemini").resolved_mt_model == "gemini-3.5-flash"
    s = load_settings(env={}, mt_provider="gemini", mt_model="gemini-3.8-flash")
    assert s.resolved_mt_model == "gemini-3.8-flash"


def test_unknown_mt_provider_rejected_at_load():
    with pytest.raises(ConfigError, match="unknown MT provider"):
        load_settings(env={}, mt_provider="bogus")


def test_gemini_key_from_env_and_require():
    s = load_settings(env={"GEMINI_API_KEY": "gk"})
    assert s.gemini_api_key == "gk"
    assert s.require_gemini_key() == "gk"
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        load_settings(env={}).require_gemini_key()


def test_settings_is_frozen():
    s = load_settings(env={})
    with pytest.raises(Exception):
        s.api_key = "x"  # type: ignore[misc]


def test_validate_chunk_bounds_accepts_cli_defaults():
    validate_chunk_bounds(5.0, 28.0)


@pytest.mark.parametrize("mins,maxs", [
    (0.0, 28.0),      # min not positive
    (-1.0, 28.0),     # min negative
    (5.0, 4.0),       # max below min
    (5.0, 5.0),       # max == min
    (5.0, 30.0),      # max at the Sarvam ceiling
    (5.0, 40.0),      # max above the ceiling
    (5.0, 9.0),       # max < 2x min
])
def test_validate_chunk_bounds_rejects(mins, maxs):
    with pytest.raises(ConfigError):
        validate_chunk_bounds(mins, maxs)


def test_validate_chunk_bounds_message_keeps_cli_wording():
    # tests/test_cli_live.py asserts on this substring, so the wording is load-bearing.
    with pytest.raises(ConfigError) as exc:
        validate_chunk_bounds(5.0, 40.0)
    assert "--max-chunk-s must be < 30" in str(exc.value)


def test_validate_chunk_bounds_full_message():
    # Pin the exact full message string so a future text change fails a test.
    expected = "--max-chunk-s must be < 30 and > --min-chunk-s, and at least 2x --min-chunk-s"
    with pytest.raises(ConfigError) as exc:
        validate_chunk_bounds(5.0, 40.0)
    assert str(exc.value) == expected


def test_validate_target_s_accepts_value_inside_bounds():
    validate_target_s(8.0, 5.0, 28.0)
    validate_target_s(5.0, 5.0, 28.0)
    validate_target_s(28.0, 5.0, 28.0)


@pytest.mark.parametrize("target", [4.9, 28.1])
def test_validate_target_s_rejects_outside_bounds(target):
    with pytest.raises(ConfigError):
        validate_target_s(target, 5.0, 28.0)


def test_validate_target_s_full_message():
    # Pin the exact full message string so a future text change fails a test.
    expected = "--target-s must be between --min-chunk-s and --max-chunk-s"
    with pytest.raises(ConfigError) as exc:
        validate_target_s(4.9, 5.0, 28.0)
    assert str(exc.value) == expected
