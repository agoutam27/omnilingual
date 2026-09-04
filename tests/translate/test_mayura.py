import json

import httpx
import pytest
import respx

from omnilingual.config import load_settings
from omnilingual.http import SarvamError
from omnilingual.translate.mayura import (
    MAYURA_LANGS,
    SARVAM_TRANSLATE_LANGS,
    MayuraTranslator,
    split_text,
)

MT_URL = "https://api.sarvam.ai/translate"


@pytest.fixture
def settings():
    return load_settings(api_key="test-key", env={})


def test_split_text_short_is_single_piece():
    assert split_text("नमस्ते। कैसे हो?", 1000) == ["नमस्ते। कैसे हो?"]


def test_split_text_breaks_on_sentence_boundaries():
    text = "पहला वाक्य। दूसरा वाक्य। तीसरा वाक्य।"
    pieces = split_text(text, 25)
    assert pieces == ["पहला वाक्य। दूसरा वाक्य।", "तीसरा वाक्य।"] or all(len(p) <= 25 for p in pieces)
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")


def test_split_text_hard_splits_when_no_boundary():
    text = "x" * 2500
    pieces = split_text(text, 1000)
    assert [len(p) for p in pieces] == [1000, 1000, 500]


def test_split_text_handles_latin_punctuation():
    text = "First sentence. Second one? Third!"
    pieces = split_text(text, 20)
    assert all(len(p) <= 20 for p in pieces)
    assert " ".join(pieces) == text


def test_language_sets():
    assert {"hi-IN", "ta-IN", "en-IN"} <= MAYURA_LANGS
    assert "ur-IN" not in MAYURA_LANGS
    assert "ur-IN" in SARVAM_TRANSLATE_LANGS
    assert MAYURA_LANGS < SARVAM_TRANSLATE_LANGS


def test_supports_depends_on_model(settings):
    assert MayuraTranslator(settings).supports("hi-IN")
    assert not MayuraTranslator(settings).supports("ur-IN")
    assert not MayuraTranslator(settings).supports("unknown")
    s2 = load_settings(api_key="k", env={}, mt_model="sarvam-translate:v1")
    assert MayuraTranslator(s2).supports("ur-IN")


@respx.mock
def test_to_english_sends_json_and_parses(settings):
    route = respx.post(MT_URL).mock(
        return_value=httpx.Response(
            200, json={"request_id": "r", "translated_text": "Hello everyone", "source_language_code": "hi-IN"}
        )
    )
    out = MayuraTranslator(settings, sleep=lambda s: None).to_english("नमस्ते सब लोग", "hi-IN")
    assert out == "Hello everyone"
    req = route.calls.last.request
    assert req.headers["api-subscription-key"] == "test-key"
    assert json.loads(req.content) == {
        "input": "नमस्ते सब लोग",
        "source_language_code": "hi-IN",
        "target_language_code": "en-IN",
        "model": "mayura:v1",
        "mode": "formal",
    }


@respx.mock
def test_to_english_splits_long_text_and_joins(settings):
    route = respx.post(MT_URL)
    route.side_effect = lambda request: httpx.Response(
        200, json={"translated_text": "EN(" + json.loads(request.content)["input"][:3] + ")"}
    )
    text = ("क" * 600 + "। ") * 3  # ~1806 chars, three sentences
    out = MayuraTranslator(settings, sleep=lambda s: None).to_english(text, "hi-IN")
    assert route.call_count >= 2
    assert out.count("EN(") == route.call_count
    for call in route.calls:
        assert len(json.loads(call.request.content)["input"]) <= 1000


@respx.mock
def test_retries_on_5xx(settings):
    route = respx.post(MT_URL)
    route.side_effect = [httpx.Response(502), httpx.Response(200, json={"translated_text": "ok"})]
    assert MayuraTranslator(settings, sleep=lambda s: None).to_english("x", "ta-IN") == "ok"
    assert route.call_count == 2


@respx.mock
def test_non_json_body_raises_sarvam_error(settings):
    respx.post(MT_URL).mock(
        return_value=httpx.Response(200, text="<html>gateway</html>", headers={"content-type": "text/html"})
    )
    with pytest.raises(SarvamError) as ei:
        MayuraTranslator(settings, sleep=lambda s: None).to_english("नमस्ते", "hi-IN")
    assert "non-JSON response body" in str(ei.value)
    assert ei.value.status == 200
    assert "gateway" in ei.value.body
