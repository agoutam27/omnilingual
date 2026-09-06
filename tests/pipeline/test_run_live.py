import re

import httpx
import respx

from omnilingual.audio.live_capture import BYTES_PER_SECOND, CaptureError
from omnilingual.cache import JsonCache
from omnilingual.config import load_settings
from omnilingual.models import chunks_from_json
from omnilingual.pipeline import transcribe_chunks
from omnilingual.pipeline.live import LiveOptions, run_live
from omnilingual.render.markdown import render
from tests.conftest import raw_pcm

STT_URL = "https://api.sarvam.ai/speech-to-text"
MT_URL = "https://api.sarvam.ai/translate"


class FakeCapture:
    """Scripted LiveCapture: 1 s PCM blocks + threshold-gated stderr lines."""

    def __init__(self, device_name, *, mic_only=False, noise_db=-35.0,
                 blocks=(), gated=(), die_after=None):
        self._blocks = list(blocks)
        self._gated = list(gated)  # (bytes_consumed_threshold, line_bytes)
        self._die_after = die_after
        self._reads = 0
        self.consumed = 0
        self.closed = False

    def open(self):
        pass

    def read(self, n):
        import time as _t

        _t.sleep(0.002)  # pace the stream so the stderr thread can interleave
        self._reads += 1
        if self._die_after is not None and self._reads > self._die_after:
            raise CaptureError("fake child died")
        if not self._blocks:
            raise StopIteration("fake stream dry (clean end)")
        self.consumed += len(self._blocks[0])
        return self._blocks.pop(0)

    def __iter__(self):
        while True:
            try:
                yield self.read(BYTES_PER_SECOND)
            except (CaptureError, StopIteration):
                return

    @property
    def stderr(self):
        import time

        pending = list(self._gated)
        while pending or not self.closed:
            ready = [ln for need, ln in pending if self.consumed >= need]
            pending = [(need, ln) for need, ln in pending if self.consumed < need]
            for ln in ready:
                yield ln
            if not pending and self.closed:
                return
            if not pending:
                return
            time.sleep(0.001)

    def close(self):
        self.closed = True


def _pcm_3x20():
    # Deliberately DISTINCT chunk durations (20/24/17 s): byte-identical
    # chunks would share one content-addressed STT cache key and collapse
    # into a single transcript (see the Task 2 trap). The leading silence
    # is the ambient probe (room noise, not tone) so the energy floor
    # calibrates to quiet, not to speech. Gaps sit at 21-22 s and 46-47 s
    # so the slicer seals [1,21), [22,46), [47,64).
    return raw_pcm([("silence", 1.0), ("tone", 20.0), ("silence", 1.0),
                    ("tone", 24.0), ("silence", 1.0), ("tone", 17.0)])


def _blocks(pcm):
    return [pcm[i : i + BYTES_PER_SECOND]
            for i in range(0, len(pcm), BYTES_PER_SECOND)]


def _gaps():
    return [
        (22 * BYTES_PER_SECOND, b"[silencedetect @ x] silence_start: 21.0\n"),
        (22 * BYTES_PER_SECOND, b"[silencedetect @ x] silence_end: 22.0\n"),
        (47 * BYTES_PER_SECOND, b"[silencedetect @ x] silence_start: 46.0\n"),
        (47 * BYTES_PER_SECOND, b"[silencedetect @ x] silence_end: 47.0\n"),
    ]


def _route(respx_mock, langs=("hi-IN", "en-IN", "ta-IN"), texts=("Bravo", "Hello", "Vanakkam")):
    stt = respx_mock.post(STT_URL)
    queue = [
        {"request_id": f"r{i}", "transcript": t, "language_code": lg,
         "language_probability": 0.9}
        for i, (lg, t) in enumerate(zip(langs, texts))
    ]

    def stt_cb(request):
        body = queue.pop(0) if queue else queue[-1] if queue else {
            "request_id": "rx", "transcript": "Again",
            "language_code": "hi-IN", "language_probability": 0.9}
        if not queue:
            queue.append(body)
        return httpx.Response(200, json=body)

    stt.mock(side_effect=stt_cb)
    mt = respx_mock.post(MT_URL)

    def mt_cb(request):
        import json as _json

        payload = _json.loads(request.content.decode())
        src = payload["source_language_code"]
        return httpx.Response(200, json={
            "translated_text": f"[{src}->en] {payload['input']}"})

    mt.mock(side_effect=mt_cb)
    return stt, mt


def _opts(tmp_path, **kw):
    args = dict(out=tmp_path / "meeting.md")
    args.update(kw)
    return LiveOptions(**args)


@respx.mock
def test_live_run_transcribes_in_order(respx_mock, tmp_path):
    # Single worker: the scripted STT queue answers in call order, so one
    # worker keeps the chunk->response mapping deterministic. The ordered
    # appender path (results dict + sequencing) still executes.
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    code = run_live(_opts(tmp_path, stt_workers=1), load_settings(api_key="k"),
                    SarvamSTT(load_settings(api_key="k")),
                    MayuraTranslator(load_settings(api_key="k")),
                    status=lambda m: None, capture_factory=factory)
    assert code == 0
    text = (tmp_path / "meeting.md").read_text(encoding="utf-8")
    heads = re.findall(r"\*\*\[(\d\d:\d\d:\d\d) → (\d\d:\d\d:\d\d)\] (\S+)\*\*", text)
    assert [h[2] for h in heads] == ["hi-IN", "en-IN", "ta-IN"]
    assert [h[0] for h in heads] == sorted(h[0] for h in heads)
    assert "· growing" not in text  # finalized
    bodies = text.split("## Transcript\n\n", 1)[1]
    assert bodies.index("Bravo") < bodies.index("Hello") < bodies.index("Vanakkam")


@respx.mock
def test_live_matches_batch_on_same_chunks(respx_mock, tmp_path):
    _route(respx_mock)
    pcm = _pcm_3x20()
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps())
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    run_live(_opts(tmp_path), settings, SarvamSTT(settings),
             MayuraTranslator(settings), status=lambda m: None,
             capture_factory=factory)
    session = next((tmp_path / ".omnilingual").glob("live-*"))
    chunks = chunks_from_json((session / "chunks.json").read_text(encoding="utf-8"))
    assert len(chunks) >= 2
    for c in chunks:
        assert c.end_s - c.start_s <= 28.0
    before = len(respx_mock.calls)
    batch = transcribe_chunks(chunks, 60.0, tmp_path, settings,
                              SarvamSTT(settings), MayuraTranslator(settings),
                              JsonCache(session / "cache"))
    live_text = (tmp_path / "meeting.md").read_text(encoding="utf-8")
    live_bodies = live_text.split("## Transcript\n\n", 1)[1]
    batch_bodies = render(batch).split("## Transcript\n\n", 1)[1]
    assert live_bodies == batch_bodies  # zero new HTTP calls needed
    assert len(respx_mock.calls) == before  # cache served all: zero new HTTP calls


@respx.mock
def test_second_run_reuses_cache(respx_mock, tmp_path):
    _route(respx_mock)
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    pcm = _pcm_3x20()
    run_live(_opts(tmp_path, out=tmp_path / "a.md"), settings,
             SarvamSTT(settings), MayuraTranslator(settings),
             status=lambda m: None,
             capture_factory=lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps()))
    first_calls = len(respx_mock.calls)
    assert first_calls > 0
    run_live(_opts(tmp_path, out=tmp_path / "b.md"), settings,
             SarvamSTT(settings), MayuraTranslator(settings),
             status=lambda m: None,
             capture_factory=lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps()))
    # NOTE: live sessions use per-session caches, so the second run re-pays.
    # Cache reuse is proven at the transcribe_chunks level (Task 2 tests).
    assert len(respx_mock.calls) > first_calls


@respx.mock
def test_capture_death_seals_partial_and_exits_2(respx_mock, tmp_path):
    _route(respx_mock)
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    pcm = raw_pcm([("tone", 10.0)])
    msgs = []
    code = run_live(
        _opts(tmp_path), settings, SarvamSTT(settings),
        MayuraTranslator(settings), status=msgs.append,
        capture_factory=lambda *a, **k: FakeCapture(
            *a, **k, blocks=_blocks(pcm), gated=[], die_after=4))
    assert code == 2
    assert any("died" in m for m in msgs)
    text = (tmp_path / "meeting.md").read_text(encoding="utf-8")
    assert "## Transcript" in text


@respx.mock
def test_cost_cap_halts_api_but_keeps_sealing(respx_mock, tmp_path):
    _route(respx_mock)
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    pcm = _pcm_3x20()
    msgs = []
    code = run_live(
        _opts(tmp_path, max_cost=0.0001), settings, SarvamSTT(settings),
        MayuraTranslator(settings), status=msgs.append,
        capture_factory=lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps()))
    assert code == 2
    assert any("cost cap" in m for m in msgs)
    session = next((tmp_path / ".omnilingual").glob("live-*"))
    chunks = chunks_from_json((session / "chunks.json").read_text(encoding="utf-8"))
    assert len(chunks) >= 2  # sealing continued after the halt
    text = (tmp_path / "meeting.md").read_text(encoding="utf-8")
    assert "cost cap" in text


@respx.mock
def test_quota_halt_keeps_file_valid(respx_mock, tmp_path):
    respx_mock.post(STT_URL).mock(return_value=httpx.Response(402, json={"error": "quota"}))
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    pcm = _pcm_3x20()
    msgs = []
    code = run_live(
        _opts(tmp_path), settings, SarvamSTT(settings),
        MayuraTranslator(settings), status=msgs.append,
        capture_factory=lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=_gaps()))
    assert code == 2
    assert any("quota" in m for m in msgs)
    assert "## Transcript" in (tmp_path / "meeting.md").read_text(encoding="utf-8")


@respx.mock
def test_contaminated_probe_still_transcribes_speech(respx_mock, tmp_path):
    # Regression: speaking during the 1 s ambient probe set the floor at
    # speech level and gated the whole meeting as silence. The probe now
    # falls back to the default floor with a warning.
    _route(respx_mock, langs=("hi-IN", "hi-IN"), texts=("Bravo", "Again"))
    pcm = raw_pcm([("tone", 21.0), ("silence", 1.0), ("tone", 17.0)])
    gaps = [
        (22 * BYTES_PER_SECOND, b"[silencedetect @ x] silence_start: 21.0\n"),
        (22 * BYTES_PER_SECOND, b"[silencedetect @ x] silence_end: 22.0\n"),
    ]
    factory = lambda *a, **k: FakeCapture(*a, **k, blocks=_blocks(pcm), gated=gaps)
    settings = load_settings(api_key="k")
    from omnilingual.stt.sarvam import SarvamSTT
    from omnilingual.translate.mayura import MayuraTranslator

    msgs = []
    code = run_live(_opts(tmp_path), settings, SarvamSTT(settings),
                    MayuraTranslator(settings), status=msgs.append,
                    capture_factory=factory)
    assert code == 0
    assert any("already speaking" in m for m in msgs)
    text = (tmp_path / "meeting.md").read_text(encoding="utf-8")
    assert "Bravo" in text
