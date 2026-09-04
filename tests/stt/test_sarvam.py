from pathlib import Path

import httpx
import pytest
import respx

from omnilingual.config import load_settings
from omnilingual.http import AuthError, SarvamError
from omnilingual.stt.sarvam import SarvamSTT

STT_URL = "https://api.sarvam.ai/speech-to-text"


@pytest.fixture
def settings():
    return load_settings(api_key="test-key", env={})


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    p = tmp_path / "c.wav"
    p.write_bytes(b"RIFF....WAVEfake")
    return p


@respx.mock
def test_transcribe_sends_multipart_and_parses(settings, wav):
    route = respx.post(STT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "request_id": "r1",
                "transcript": "  नमस्ते सब लोग  ",
                "language_code": "hi-IN",
                "language_probability": 0.97,
            },
        )
    )
    result = SarvamSTT(settings, sleep=lambda s: None).transcribe(wav)

    assert result.text == "नमस्ते सब लोग"
    assert result.lang == "hi-IN"
    assert result.prob == 0.97

    req = route.calls.last.request
    assert req.headers["api-subscription-key"] == "test-key"
    ct = req.headers["content-type"]
    assert ct.startswith("multipart/form-data")
    body = req.content
    assert b'name="model"' in body and b"saaras:v4" in body
    assert b'name="mode"' in body and b"transcribe" in body
    assert b'name="language_code"' in body and b"unknown" in body
    assert b'filename="c.wav"' in body
    assert b"RIFF....WAVEfake" in body


@respx.mock
def test_null_language_fields_default(settings, wav):
    respx.post(STT_URL).mock(
        return_value=httpx.Response(
            200, json={"transcript": "hello", "language_code": None, "language_probability": None}
        )
    )
    r = SarvamSTT(settings, sleep=lambda s: None).transcribe(wav)
    assert r.lang == "unknown"
    assert r.prob == 0.0
    assert r.text == "hello"


@respx.mock
def test_retries_then_succeeds(settings, wav):
    route = respx.post(STT_URL)
    route.side_effect = [
        httpx.Response(429),
        httpx.Response(200, json={"transcript": "ok", "language_code": "en-IN", "language_probability": 1.0}),
    ]
    r = SarvamSTT(settings, sleep=lambda s: None).transcribe(wav)
    assert r.text == "ok"
    assert route.call_count == 2


@respx.mock
def test_auth_error_propagates(settings, wav):
    respx.post(STT_URL).mock(return_value=httpx.Response(401, text="nope"))
    with pytest.raises(AuthError):
        SarvamSTT(settings, sleep=lambda s: None).transcribe(wav)


def test_uses_settings_model_and_base_url(wav):
    s = load_settings(api_key="k", env={}, stt_model="saaras:v3", base_url="https://alt.example")
    with respx.mock:
        route = respx.post("https://alt.example/speech-to-text").mock(
            return_value=httpx.Response(200, json={"transcript": "x", "language_code": "ta-IN", "language_probability": 0.5})
        )
        SarvamSTT(s, sleep=lambda t: None).transcribe(wav)
        assert b"saaras:v3" in route.calls.last.request.content


@respx.mock
def test_non_json_body_raises_sarvam_error(settings, wav):
    respx.post(STT_URL).mock(
        return_value=httpx.Response(200, text="<html>gateway</html>", headers={"content-type": "text/html"})
    )
    with pytest.raises(SarvamError) as ei:
        SarvamSTT(settings, sleep=lambda s: None).transcribe(wav)
    assert "non-JSON response body" in str(ei.value)
    assert ei.value.status == 200
    assert "gateway" in ei.value.body
