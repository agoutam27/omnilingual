import pytest

from omnilingual.cache import JsonCache
from omnilingual.config import load_settings
from omnilingual.http import SarvamError, TransientError
from omnilingual.models import Chunk, STTResult
from omnilingual.pipeline.live import calibrate_energy_floor, process_chunk, resolve_energy_floor
from tests.conftest import make_wav, raw_pcm


class FakeSTT:
    model = "fake-stt"
    mode = "transcribe"

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def transcribe(self, wav_path):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class FakeTranslator:
    model = "fake-mt"

    def __init__(self, supported=("hi-IN", "en-IN"), error=None):
        self.supported = set(supported)
        self.error = error
        self.calls = 0

    def supports(self, lang):
        return lang in self.supported

    def to_english(self, text, src_lang):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return f"[{src_lang}->en] {text}"


def _ctx(tmp_path, **kw):
    wav = tmp_path / "c.wav"
    make_wav(wav, [("tone", 2.0)])
    chunk = Chunk(idx=0, start_s=0.0, end_s=2.0, wav_path=wav)
    settings = load_settings(api_key="k")
    cache = JsonCache(tmp_path / "cache")
    return dict(chunk=chunk, cache=cache, settings=settings, **kw)


def test_ok_hindi_bills_stt_plus_mt(tmp_path):
    stt = FakeSTT(STTResult(lang="hi-IN", prob=0.9, text="Namaste"))
    tr = FakeTranslator()
    ctx = _ctx(tmp_path, speech=True, stt=stt, translator=tr)
    seg, delta, billable = process_chunk(**ctx)
    assert (seg.status, seg.english) == ("ok", "[hi-IN->en] Namaste")
    assert billable is True and delta > 0
    assert stt.calls == 1 and tr.calls == 1


def test_english_skips_translator(tmp_path):
    stt = FakeSTT(STTResult(lang="en-IN", prob=0.99, text="Hello"))
    tr = FakeTranslator()
    ctx = _ctx(tmp_path, speech=True, stt=stt, translator=tr)
    seg, delta, billable = process_chunk(**ctx)
    assert seg.status == "ok" and seg.english is None and tr.calls == 0


def test_unsupported_language_marks_mt_unsupported(tmp_path):
    stt = FakeSTT(STTResult(lang="xx-YY", prob=0.8, text="lą"))
    tr = FakeTranslator(supported=("hi-IN",))
    ctx = _ctx(tmp_path, speech=True, stt=stt, translator=tr)
    seg, delta, billable = process_chunk(**ctx)
    assert seg.status == "mt_unsupported" and seg.english is None


def test_transient_mt_failure_marks_mt_failed(tmp_path):
    stt = FakeSTT(STTResult(lang="hi-IN", prob=0.9, text="Namaste"))
    tr = FakeTranslator(error=TransientError("overloaded", status=503))
    ctx = _ctx(tmp_path, speech=True, stt=stt, translator=tr)
    seg, delta, billable = process_chunk(**ctx)
    assert seg.status == "mt_failed"


def test_stt_error_marks_stt_failed(tmp_path):
    stt = FakeSTT(error=SarvamError("boom"))
    tr = FakeTranslator()
    ctx = _ctx(tmp_path, speech=True, stt=stt, translator=tr)
    seg, delta, billable = process_chunk(**ctx)
    assert seg.status == "stt_failed" and seg.text == "[transcription failed]"


def test_empty_text_is_no_speech(tmp_path):
    stt = FakeSTT(STTResult(lang="hi-IN", prob=0.9, text="  "))
    tr = FakeTranslator()
    ctx = _ctx(tmp_path, speech=True, stt=stt, translator=tr)
    seg, delta, billable = process_chunk(**ctx)
    assert seg.status == "no_speech" and tr.calls == 0


def test_silent_chunk_makes_no_api_calls(tmp_path):
    stt = FakeSTT(STTResult(lang="hi-IN", prob=0.9, text="Namaste"))
    tr = FakeTranslator()
    ctx = _ctx(tmp_path, speech=False, stt=stt, translator=tr)
    seg, delta, billable = process_chunk(**ctx)
    assert seg.status == "no_speech" and (delta, billable) == (0.0, False)
    assert stt.calls == 0 and tr.calls == 0


def test_calibrate_energy_floor():
    assert calibrate_energy_floor(0.001) == 0.004
    assert calibrate_energy_floor(0.005) == pytest.approx(0.02)


def test_resolve_energy_floor_quiet_probe_calibrates():
    from omnilingual.audio.live_slicer import DEFAULT_ENERGY_FLOOR

    floor, contaminated = resolve_energy_floor(b"\x00" * 32000)
    assert contaminated is False
    assert floor == DEFAULT_ENERGY_FLOOR


def test_resolve_energy_floor_voice_probe_falls_back_with_flag():
    from omnilingual.audio.live_slicer import DEFAULT_ENERGY_FLOOR

    floor, contaminated = resolve_energy_floor(raw_pcm([("tone", 1.0)]))
    assert contaminated is True
    assert floor == DEFAULT_ENERGY_FLOOR
