import logging
from pathlib import Path

import pytest

from omnilingual.cache import JsonCache
from omnilingual.config import load_settings
from omnilingual.http import QuotaError, TransientError
from omnilingual.models import STTResult
from omnilingual.pipeline import estimate, prepare, run, work_dir_for
from tests.conftest import make_wav, requires_ffmpeg


class FakeSTT:
    model = "fake-stt"
    mode = "transcribe"

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def transcribe(self, wav_path: Path) -> STTResult:
        self.calls += 1
        r = self._results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeMT:
    model = "fake-mt"

    def __init__(self, fail_on=()):
        self.calls = 0
        self._fail_on = set(fail_on)

    def supports(self, lang: str) -> bool:
        return lang in {"hi-IN", "ta-IN", "en-IN"}

    def to_english(self, text: str, src_lang: str) -> str:
        self.calls += 1
        if text in self._fail_on:
            raise TransientError("mt down", 503)
        return f"EN[{text}]"


@pytest.fixture
def settings():
    return load_settings(api_key="k", env={}, max_chunk_s=10.0, min_chunk_s=2.0, langs=["hi-IN", "en-IN"])


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    # 3 chunks: 10 + 10 + 5 seconds (tones with tiny gaps that are below silence threshold
    # length). The gaps also keep each 10s tone from being byte-identical to the next: at
    # 440Hz, a bare continuous 10s tone repeats exactly every 10s (440*10 is a whole number
    # of cycles), which would make chunk 0 and chunk 1 collide under content-addressed caching.
    return make_wav(
        tmp_path / "meeting.wav",
        [("tone", 10.0), ("silence", 0.05), ("tone", 9.95), ("silence", 0.05), ("tone", 4.95)],
    )


def test_work_dir_for_is_content_addressed(tmp_path: Path):
    a = tmp_path / "a.bin"; a.write_bytes(b"same")
    b = tmp_path / "b.bin"; b.write_bytes(b"same")
    c = tmp_path / "c.bin"; c.write_bytes(b"different")
    root = tmp_path / "work"
    assert work_dir_for(a, root) == work_dir_for(b, root)
    assert work_dir_for(a, root) != work_dir_for(c, root)
    assert work_dir_for(a, root).parent == root
    assert len(work_dir_for(a, root).name) == 12


@requires_ffmpeg
def test_prepare_creates_and_reuses_chunks(tmp_path: Path, settings, recording):
    wd = tmp_path / "wd"
    duration, chunks = prepare(recording, wd, settings)
    assert abs(duration - 25.0) < 0.1
    assert (wd / "normalized.wav").exists()
    assert (wd / "chunks.json").exists()
    assert len(chunks) == 3
    assert all(c.wav_path.exists() for c in chunks)
    mtime = (wd / "chunks.json").stat().st_mtime
    duration2, chunks2 = prepare(recording, wd, settings)
    assert chunks2 == chunks
    assert (wd / "chunks.json").stat().st_mtime == mtime


def test_estimate_prices_audio_and_chars(settings):
    from omnilingual.models import Chunk
    chunks = [Chunk(0, 0, 60.0, Path("x")), Chunk(1, 60.0, 120.0, Path("y"))]
    cost = estimate(120.0, chunks, settings, chars_per_second=10.0)
    assert cost.audio_seconds == 120.0
    assert cost.mt_chars == 1200
    # 120s of ₹30/hr = ₹1.00 ; 1200 chars of ₹20/10k = ₹2.40
    assert abs(cost.inr_estimate - 3.40) < 1e-6


@requires_ffmpeg
def test_run_happy_path_and_statuses(tmp_path: Path, settings, recording, caplog):
    wd = tmp_path / "wd"
    stt = FakeSTT([
        STTResult("hi-IN", 0.95, "नमस्ते"),
        STTResult("en-IN", 0.99, "hello"),
        STTResult("ta-IN", 0.90, "வணக்கம்"),   # outside settings.langs → warning, still translated
    ])
    mt = FakeMT()
    with caplog.at_level(logging.WARNING):
        t = run(recording, wd, settings, stt, mt, JsonCache(wd / "cache"))

    assert [s.lang for s in t.segments] == ["hi-IN", "en-IN", "ta-IN"]
    assert t.segments[0].english == "EN[नमस्ते]" and t.segments[0].status == "ok"
    assert t.segments[1].english is None and t.segments[1].status == "ok"
    assert t.segments[2].english == "EN[வணக்கம்]"
    assert mt.calls == 2  # en-IN skipped
    assert "ta-IN" in caplog.text
    assert t.cost.audio_seconds == pytest.approx(25.0, abs=0.1)
    assert t.cost.mt_chars == len("नमस्ते") + len("வணக்கம்")
    assert t.source == recording


@requires_ffmpeg
def test_run_second_time_uses_cache(tmp_path: Path, settings, recording):
    wd = tmp_path / "wd"
    cache = JsonCache(wd / "cache")
    stt1 = FakeSTT([STTResult("hi-IN", 0.9, "क"), STTResult("hi-IN", 0.9, "ख"), STTResult("hi-IN", 0.9, "ग")])
    mt1 = FakeMT()
    run(recording, wd, settings, stt1, mt1, cache)
    stt2 = FakeSTT([])
    mt2 = FakeMT()
    t = run(recording, wd, settings, stt2, mt2, cache)
    assert stt2.calls == 0 and mt2.calls == 0
    assert [s.text for s in t.segments] == ["क", "ख", "ग"]


@requires_ffmpeg
def test_run_marks_failed_chunks_and_continues(tmp_path: Path, settings, recording):
    wd = tmp_path / "wd"
    stt = FakeSTT([
        TransientError("stt down", 503),
        STTResult("hi-IN", 0.9, "fail-me"),
        STTResult("kn-IN", 0.8, "ಕನ್ನಡ"),   # FakeMT does not support kn-IN
    ])
    mt = FakeMT(fail_on={"fail-me"})
    t = run(recording, wd, settings, stt, mt, JsonCache(wd / "cache"))
    assert [s.status for s in t.segments] == ["stt_failed", "mt_failed", "mt_unsupported"]
    assert t.segments[0].text == "[transcription failed]"
    assert t.segments[1].english is None
    assert t.segments[2].english is None


@requires_ffmpeg
def test_run_quota_error_propagates_after_caching(tmp_path: Path, settings, recording):
    wd = tmp_path / "wd"
    cache = JsonCache(wd / "cache")
    stt = FakeSTT([STTResult("hi-IN", 0.9, "क"), QuotaError("out of credits", 402)])
    with pytest.raises(QuotaError):
        run(recording, wd, settings, stt, FakeMT(), cache)
    # resume: only the remaining two chunks hit STT
    stt2 = FakeSTT([STTResult("hi-IN", 0.9, "ख"), STTResult("hi-IN", 0.9, "ग")])
    t = run(recording, wd, settings, stt2, FakeMT(), cache)
    assert stt2.calls == 2
    assert [s.text for s in t.segments] == ["क", "ख", "ग"]


@requires_ffmpeg
def test_progress_callback(tmp_path: Path, settings, recording):
    seen = []
    stt = FakeSTT([STTResult("en-IN", 1.0, "a"), STTResult("en-IN", 1.0, "b"), STTResult("en-IN", 1.0, "c")])
    run(recording, tmp_path / "wd", settings, stt, FakeMT(), JsonCache(tmp_path / "c"),
        progress=lambda i, n, seg: seen.append((i, n, seg.text)))
    assert seen == [(1, 3, "a"), (2, 3, "b"), (3, 3, "c")]
