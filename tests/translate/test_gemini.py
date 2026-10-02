import json

import httpx
import pytest
import respx

from omnilingual.config import ConfigError, load_settings
from omnilingual.http import SarvamError
from omnilingual.translate.gemini import LANG_NAMES, GeminiTranslator

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"


@pytest.fixture
def settings():
    return load_settings(env={"GEMINI_API_KEY": "gk"}, mt_provider="gemini")


def _ok(text: str) -> httpx.Response:
    return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})


def test_prices_as_free():
    assert GeminiTranslator(load_settings(env={"GEMINI_API_KEY": "gk"}, mt_provider="gemini")).inr_per_10k_chars == 0.0


def test_supports_all_indic_languages_plus_english():
    assert {"hi-IN", "ta-IN", "ur-IN", "od-IN", "en-IN"} <= LANG_NAMES.keys()
    tr = GeminiTranslator(load_settings(env={}, mt_provider="gemini"))
    assert tr.supports("hi-IN")
    assert tr.supports("ur-IN")
    assert not tr.supports("unknown")
    assert not tr.supports("hi")


@respx.mock
def test_to_english_sends_key_and_parses(settings):
    route = respx.post(GEMINI_URL).mock(return_value=_ok("Hello everyone"))
    assert GeminiTranslator(settings, sleep=lambda s: None).to_english("नमस्ते सब लोग", "hi-IN") == "Hello everyone"
    req = route.calls.last.request
    assert req.headers["x-goog-api-key"] == "gk"
    body = json.loads(req.content)
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Language: Hindi\n\nनमस्ते सब लोग"}]}]
    assert body["systemInstruction"]["parts"][0]["text"].startswith("You are a translation engine")
    assert body["generationConfig"] == {"temperature": 0.0}


@respx.mock
def test_unknown_lang_falls_back_to_raw_tag(settings):
    route = respx.post(GEMINI_URL).mock(return_value=_ok("ok"))
    GeminiTranslator(settings, sleep=lambda s: None).to_english("x", "xx-YY")
    assert json.loads(route.calls.last.request.content)["contents"][0]["parts"][0]["text"] == "Language: xx-YY\n\nx"


@respx.mock
def test_joins_multiple_text_parts(settings):
    respx.post(GEMINI_URL).mock(return_value=httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": "Hello "}, {"text": "there"}]}}]}))
    assert GeminiTranslator(settings, sleep=lambda s: None).to_english("नमस्ते", "hi-IN") == "Hello there"


@respx.mock
def test_thought_parts_without_text_are_skipped(settings):
    respx.post(GEMINI_URL).mock(return_value=httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": None}, {"text": "Hello"}]}}]}))
    assert GeminiTranslator(settings, sleep=lambda s: None).to_english("नमस्ते", "hi-IN") == "Hello"


@respx.mock
def test_splits_long_text_and_joins(settings):
    route = respx.post(GEMINI_URL)
    route.side_effect = lambda request: _ok("EN(" + request.read().decode()[:0] + ")")
    text = ("क" * 1500 + "। ") * 3  # 4506 chars, three sentences
    out = GeminiTranslator(settings, sleep=lambda s: None).to_english(text, "hi-IN")
    assert route.call_count >= 2
    assert out.count("EN(") == route.call_count
    for call in route.calls:
        payload = json.loads(call.request.content)["contents"][0]["parts"][0]["text"]
        assert len(payload) <= 4000


@respx.mock
def test_retries_on_5xx(settings):
    route = respx.post(GEMINI_URL)
    route.side_effect = [httpx.Response(502), _ok("ok")]
    assert GeminiTranslator(settings, sleep=lambda s: None).to_english("x", "ta-IN") == "ok"
    assert route.call_count == 2


@respx.mock
def test_bad_key_reports_gemini_message(settings):
    # Gemini signals a bad key with 400, not 401/403, so it lands on the generic
    # SarvamError branch; the API's own message has to carry the diagnosis.
    respx.post(GEMINI_URL).mock(return_value=httpx.Response(400, json={"error": {"message": "API key not valid"}}))
    with pytest.raises(SarvamError, match="API key not valid") as ei:
        GeminiTranslator(settings, sleep=lambda s: None).to_english("x", "ta-IN")
    assert ei.value.status == 400


@respx.mock
def test_blocked_prompt_reports_reason(settings):
    respx.post(GEMINI_URL).mock(return_value=httpx.Response(
        200, json={"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []}))
    with pytest.raises(SarvamError, match="blocked: SAFETY"):
        GeminiTranslator(settings, sleep=lambda s: None).to_english("x", "ta-IN")


@respx.mock
def test_empty_candidates_without_reason_still_raises(settings):
    respx.post(GEMINI_URL).mock(return_value=httpx.Response(200, json={"candidates": []}))
    with pytest.raises(SarvamError, match="no translation"):
        GeminiTranslator(settings, sleep=lambda s: None).to_english("x", "ta-IN")


@respx.mock
def test_non_json_body_raises_sarvam_error(settings):
    respx.post(GEMINI_URL).mock(
        return_value=httpx.Response(200, text="<html>gateway</html>", headers={"content-type": "text/html"}))
    with pytest.raises(SarvamError) as ei:
        GeminiTranslator(settings, sleep=lambda s: None).to_english("नमस्ते", "hi-IN")
    assert "non-JSON response body" in str(ei.value)
    assert ei.value.status == 200
    assert "gateway" in ei.value.body


def test_missing_key_fails_before_any_request():
    tr = GeminiTranslator(load_settings(env={}, mt_provider="gemini"))
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        tr.to_english("x", "hi-IN")


def test_cache_key_is_namespaced_by_provider_model():
    assert GeminiTranslator(load_settings(env={}, mt_provider="gemini")).model == "gemini:gemini-3.5-flash"