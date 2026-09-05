import threading

import httpx
import respx

from omnilingual.cache import JsonCache
from omnilingual.config import load_settings
from omnilingual.models import chunks_from_json
from omnilingual.pipeline import run_from_chunks
from omnilingual.pipeline.live import LiveOptions, run_live
from omnilingual.stt.sarvam import SarvamSTT
from omnilingual.translate.mayura import MayuraTranslator
from tests.pipeline.test_run_live import FakeCapture, _blocks, _gaps, _pcm_3x20

STT_URL = "https://api.sarvam.ai/speech-to-text"
MT_URL = "https://api.sarvam.ai/translate"


@respx.mock
def test_quota_halt_then_from_chunks_finishes_free(respx_mock, tmp_path):
    quota = {"on": False}
    succeeded = []
    flip = threading.Lock()

    def stt_cb(request):
        # Locked check-and-set: exactly one STT call succeeds no matter how
        # the worker threads interleave; the rest see the flipped quota.
        # (Counting successes in-callback rather than via respx calls: a
        # halted chunk's 402 is a real STT call too, and must not be
        # mistaken for a success.)
        #
        # Once healed, the flip stays off so the recovery run pays no 402s.
        ok = httpx.Response(200, json={
            "request_id": "r", "transcript": "Bravo",
            "language_code": "hi-IN", "language_probability": 0.9})
        with flip:
            if quota.get("healed"):
                return ok
            if quota["on"]:
                return httpx.Response(402, json={"error": "quota"})
            quota["on"] = True
            succeeded.append(request)
        return ok

    def mt_cb(request):
        return httpx.Response(200, json={"translated_text": "[hi-IN->en] Bravo"})

    respx_mock.post(STT_URL).mock(side_effect=stt_cb)
    respx_mock.post(MT_URL).mock(side_effect=mt_cb)
    settings = load_settings(api_key="k")

    def stt_calls():
        return [c for c in respx_mock.calls
                if c.request.url.path == "/speech-to-text"]

    pcm = _pcm_3x20()
    out = tmp_path / "meeting.md"
    code = run_live(
        LiveOptions(out=out), settings, SarvamSTT(settings),
        MayuraTranslator(settings), status=lambda m: None,
        capture_factory=lambda *a, **k: FakeCapture(
            *a, **k, blocks=_blocks(pcm), gated=_gaps()))
    assert code == 2  # one chunk ok and cached; the rest hit the flipped quota
    stt_at_halt = len(stt_calls())
    assert len(succeeded) == 1  # the flip let exactly one STT call through
    session = next((tmp_path / ".omnilingual").glob("live-*"))
    chunks = chunks_from_json((session / "chunks.json").read_text(encoding="utf-8"))
    assert len(chunks) >= 2
    quota["on"] = False  # quota restored: finish the halted session
    quota["healed"] = True  # keep the flip off for the recovery run
    done = run_from_chunks(session, settings, SarvamSTT(settings),
                           MayuraTranslator(settings),
                           JsonCache(session / "cache"))
    assert all(s.status == "ok" for s in done.segments)
    # The chunk transcribed before the halt was NOT re-paid:
    assert len(stt_calls()) == stt_at_halt + (len(chunks) - 1)
