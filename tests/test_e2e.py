import json
from pathlib import Path

import httpx
import respx

from omnilingual.cache import JsonCache
from omnilingual.config import load_settings
from omnilingual.pipeline import run, work_dir_for
from omnilingual.render.markdown import render, render_english_only
from omnilingual.stt.sarvam import SarvamSTT
from omnilingual.translate.mayura import MayuraTranslator
from tests.conftest import make_wav, requires_ffmpeg

STT_URL = "https://api.sarvam.ai/speech-to-text"
MT_URL = "https://api.sarvam.ai/translate"

STT_SCRIPT = [
    {"transcript": "हम आज payment dashboard पर बात करेंगे।", "language_code": "hi-IN", "language_probability": 0.96},
    {"transcript": "Refund numbers look fine this week.", "language_code": "en-IN", "language_probability": 0.99},
    {"transcript": "வணக்கம், நான் தொடங்குகிறேன்.", "language_code": "ta-IN", "language_probability": 0.93},
]


def _stt_side_effect():
    it = iter(STT_SCRIPT)
    return lambda request: httpx.Response(200, json=next(it))


def _mt_side_effect(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(200, json={"translated_text": f"[{body['source_language_code']}→en] {body['input']}"})


@requires_ffmpeg
@respx.mock
def test_full_pipeline_three_languages_then_cached(tmp_path: Path):
    # 3 speech blocks separated by 1s silences → 3 chunks with max_chunk_s=28
    rec = make_wav(tmp_path / "meeting.wav",
                   [("tone", 20.0), ("silence", 1.0), ("tone", 20.0), ("silence", 1.0), ("tone", 20.0)])
    settings = load_settings(api_key="k", env={}, langs=["hi-IN", "ta-IN", "en-IN"])
    stt_route = respx.post(STT_URL).mock(side_effect=_stt_side_effect())
    mt_route = respx.post(MT_URL).mock(side_effect=_mt_side_effect)

    wd = work_dir_for(rec, tmp_path / ".omnilingual")
    cache = JsonCache(wd / "cache")
    stt = SarvamSTT(settings, sleep=lambda s: None)
    mt = MayuraTranslator(settings, sleep=lambda s: None)

    t = run(rec, wd, settings, stt, mt, cache)

    assert stt_route.call_count == 3
    assert mt_route.call_count == 2  # en-IN chunk not translated
    assert [s.lang for s in t.segments] == ["hi-IN", "en-IN", "ta-IN"]
    assert all(s.status == "ok" for s in t.segments)

    md = render(t)
    assert "**[00:00:00 → 00:00:20] hi-IN**" in md
    assert "> [hi-IN→en] हम आज payment dashboard पर बात करेंगे।" in md
    assert "Refund numbers look fine this week." in md
    assert md.count("> ") == 2
    assert "Languages: hi-IN 3" in md or "Languages: ta-IN 3" in md or "Languages: en-IN 3" in md

    en = render_english_only(t)
    assert "[ta-IN→en] வணக்கம், நான் தொடங்குகிறேன்." in en
    assert "[hi-IN→en]" in en and "Refund numbers" in en

    # second run: zero network calls, identical output
    t2 = run(rec, wd, settings, SarvamSTT(settings, sleep=lambda s: None), MayuraTranslator(settings, sleep=lambda s: None), cache)
    assert stt_route.call_count == 3
    assert mt_route.call_count == 2
    assert render(t2) == md
