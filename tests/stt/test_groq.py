from pathlib import Path

import httpx
import pytest
import respx

from omnilingual.config import ConfigError, load_settings
from omnilingual.http import AuthError, SarvamError
from omnilingual.stt.groq import GROQ_TRANSCRIPTION_URL, GroqSTT


@pytest.fixture
def settings():
    return load_settings(api_key="sarvam-k", env={"GROQ_API_KEY": "groq-k"}, stt_provider="groq")


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    p = tmp_path / "c.wav"
    p.write_bytes(b"RIFF....WAVEfake")
    return p


@respx.mock
def test_sends_openai_style_multipart_and_maps_language(settings, wav):
    route = respx.post(GROQ_TRANSCRIPTION_URL).mock(
        return_value=httpx.Response(200, json={"text": "  नमस्ते  ", "language": "hindi"})
    )
    r = GroqSTT(settings, sleep=lambda s: None).transcribe(wav)

    assert (r.text, r.lang, r.prob) == ("नमस्ते", "hi-IN", 0.0)
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer groq-k"
    body = req.content
    assert b'name="model"' in body and b"whisper-large-v3-turbo" in body
    assert b'name="response_format"' in body and b"verbose_json" in body
    assert b'filename="c.wav"' in body
    assert b"RIFF....WAVEfake" in body


@respx.mock
def test_iso_language_code_also_maps(settings, wav):
    respx.post(GROQ_TRANSCRIPTION_URL).mock(
        return_value=httpx.Response(200, json={"text": "ok", "language": "ta"})
    )
    assert GroqSTT(settings, sleep=lambda s: None).transcribe(wav).lang == "ta-IN"


def test_missing_groq_key_raises(wav):
    s = load_settings(env={}, stt_provider="groq")
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        GroqSTT(s, sleep=lambda t: None).transcribe(wav)


@respx.mock
def test_auth_error_propagates(settings, wav):
    respx.post(GROQ_TRANSCRIPTION_URL).mock(return_value=httpx.Response(401, text="bad key"))
    with pytest.raises(AuthError):
        GroqSTT(settings, sleep=lambda s: None).transcribe(wav)


@respx.mock
def test_retries_then_succeeds(settings, wav):
    route = respx.post(GROQ_TRANSCRIPTION_URL)
    route.side_effect = [
        httpx.Response(429),
        httpx.Response(200, json={"text": "ok", "language": "english"}),
    ]
    r = GroqSTT(settings, sleep=lambda s: None).transcribe(wav)
    assert (r.text, r.lang) == ("ok", "en-IN")
    assert route.call_count == 2


@respx.mock
def test_non_json_body_raises_sarvam_error(settings, wav):
    respx.post(GROQ_TRANSCRIPTION_URL).mock(
        return_value=httpx.Response(200, text="<html>gateway</html>", headers={"content-type": "text/html"})
    )
    with pytest.raises(SarvamError, match="non-JSON response body"):
        GroqSTT(settings, sleep=lambda s: None).transcribe(wav)


def test_provider_attrs_and_model_namespacing(settings):
    stt = GroqSTT(settings)
    assert stt.mode == "transcribe"
    assert stt.inr_per_hour == 3.4
    assert stt.model == "groq:whisper-large-v3-turbo"


def test_model_override():
    s = load_settings(env={"GROQ_API_KEY": "k"}, stt_provider="groq", stt_model="whisper-large-v3")
    assert GroqSTT(s).model == "groq:whisper-large-v3"
