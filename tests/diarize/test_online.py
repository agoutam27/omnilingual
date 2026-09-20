import math
import threading
from pathlib import Path

from omnilingual.config import load_settings
from omnilingual.diarize.online import OnlineSpeakerTracker


def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_first_chunk_becomes_speaker_1(tmp_path):
    tracker = OnlineSpeakerTracker(lambda p: (1.0, 0.0))
    assert tracker.assign(tmp_path / "a.wav") == "Speaker 1"


def test_near_identical_embedding_matches(tmp_path):
    tracker = OnlineSpeakerTracker(lambda p: (1.0, 0.0) if p.stem == "a" else (1.0, 0.1))
    assert tracker.assign(tmp_path / "a.wav") == "Speaker 1"
    assert tracker.assign(tmp_path / "b.wav") == "Speaker 1"


def test_orthogonal_embedding_creates_speaker_2(tmp_path):
    tracker = OnlineSpeakerTracker(lambda p: (1.0, 0.0) if p.stem == "a" else (0.0, 1.0))
    assert tracker.assign(tmp_path / "a.wav") == "Speaker 1"
    assert tracker.assign(tmp_path / "b.wav") == "Speaker 2"


def test_labels_stay_bounded(tmp_path):
    vecs = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 1.0)]
    it = iter(vecs * 20)
    tracker = OnlineSpeakerTracker(lambda p: next(it), threshold=0.99, max_speakers=2)
    labels = {tracker.assign(tmp_path / f"{i}.wav") for i in range(10)}
    assert labels == {"Speaker 1", "Speaker 2"}


def test_centroid_moves_toward_running_mean():
    e1 = _norm((1.0, 0.0))
    e2 = _norm((1.0, 0.5))
    tracker = OnlineSpeakerTracker(lambda p: e1)
    tracker.assign(Path("a.wav"))
    before = _cos(tracker._centroids[0], e2)
    tracker._extractor = lambda p: e2
    assert tracker.assign(Path("b.wav")) == "Speaker 1"
    after = _cos(tracker._centroids[0], e2)
    assert after > before


def test_assignments_are_serialized(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    done_second = threading.Event()
    calls = {"n": 0}

    def extractor(p):
        calls["n"] += 1
        if calls["n"] == 1:
            entered.set()
            release.wait(2)
        return (1.0, 0.0)

    tracker = OnlineSpeakerTracker(extractor)
    t1 = threading.Thread(target=tracker.assign, args=(tmp_path / "a.wav",))
    t1.start()
    assert entered.wait(2)

    results: list[str] = []

    def second():
        results.append(tracker.assign(tmp_path / "b.wav"))
        done_second.set()

    t2 = threading.Thread(target=second)
    t2.start()
    assert not done_second.wait(0.2)
    release.set()
    t1.join()
    t2.join()
    assert results == ["Speaker 1"]
    assert len(tracker._centroids) == 1


def test_process_chunk_sets_speaker_from_tracker(tmp_path):
    from omnilingual.cache import JsonCache
    from omnilingual.models import Chunk, STTResult
    from omnilingual.pipeline.live import process_chunk

    class FakeSTT:
        model = "fake-stt"
        mode = "transcribe"

        def transcribe(self, wav_path):
            return STTResult("hi-IN", 0.9, "नमस्ते")

    class FakeMT:
        model = "fake-mt"

        def supports(self, lang):
            return True

        def to_english(self, text, src_lang):
            return "hello"

    chunk = Chunk(0, 0.0, 8.0, tmp_path / "c.wav")
    chunk.wav_path.write_bytes(b"RIFF....WAVEfake")
    settings = load_settings(api_key="k", env={})
    cache = JsonCache(tmp_path / "cache")
    tracker = OnlineSpeakerTracker(lambda p: (1.0, 0.0))

    seg, _, _ = process_chunk(
        chunk, speech=True, stt=FakeSTT(), translator=FakeMT(),
        cache=cache, settings=settings, tracker=tracker,
    )
    assert seg.status == "ok"
    assert seg.speaker == "Speaker 1"

    seg, _, billed = process_chunk(
        chunk, speech=False, stt=FakeSTT(), translator=FakeMT(),
        cache=JsonCache(tmp_path / "cache2"), settings=settings, tracker=tracker,
    )
    assert seg.speaker is None
    assert billed is False
