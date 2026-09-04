"""Opt-in tests that hit the real Sarvam API. Run: SARVAM_API_KEY=... uv run pytest -m live"""

import os
from pathlib import Path

import pytest

from omnilingual.config import load_settings
from omnilingual.stt.sarvam import SarvamSTT
from omnilingual.translate.mayura import MayuraTranslator
from tests.conftest import make_wav

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("SARVAM_API_KEY"), reason="SARVAM_API_KEY not set"),
]


def test_live_stt_returns_language_and_text(tmp_path: Path):
    # A pure tone has no speech; we only assert the request shape is accepted (HTTP 200) and fields parse.
    wav = make_wav(tmp_path / "tone.wav", [("tone", 3.0)])
    result = SarvamSTT(load_settings()).transcribe(wav)
    assert isinstance(result.text, str)
    assert isinstance(result.lang, str)
    assert 0.0 <= result.prob <= 1.0


def test_live_translate_hindi_to_english():
    out = MayuraTranslator(load_settings()).to_english("नमस्ते, आप कैसे हैं?", "hi-IN")
    assert out
    assert any(word in out.lower() for word in ("hello", "how", "you"))
