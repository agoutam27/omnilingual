from pathlib import Path

import pytest

from omnilingual.audio.chunker import (
    Silence,
    chunk_audio,
    cut_chunks,
    detect_silences,
    plan_chunks,
)
from tests.conftest import make_wav, requires_ffmpeg

MAX, MIN = 28.0, 5.0


def _assert_tiling(spans, duration):
    assert spans[0][0] == 0.0
    assert abs(spans[-1][1] - duration) < 1e-6
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert abs(a1 - b0) < 1e-6
    for s, e in spans:
        assert e - s <= MAX + 1e-6


def test_short_audio_is_single_chunk():
    assert plan_chunks(12.0, [], MAX, MIN) == [(0.0, 12.0)]


def test_audio_shorter_than_min_is_single_chunk():
    assert plan_chunks(2.0, [], MAX, MIN) == [(0.0, 2.0)]


def test_no_silences_hard_cuts_at_max():
    spans = plan_chunks(70.0, [], MAX, MIN)
    _assert_tiling(spans, 70.0)
    assert spans[0] == (0.0, 28.0)
    assert spans[1] == (28.0, 56.0)
    assert spans[2] == (56.0, 70.0)


def test_prefers_latest_silence_in_window():
    silences = [Silence(10.0, 10.6), Silence(24.0, 24.8), Silence(40.0, 40.4)]
    spans = plan_chunks(60.0, silences, MAX, MIN)
    _assert_tiling(spans, 60.0)
    assert spans[0] == (0.0, 24.4)  # mid of (24.0, 24.8), not 10.3, not hard 28
    assert spans[1][1] == 40.2      # mid of (40.0, 40.4) within [29.4, 52.4]


def test_silence_before_min_is_ignored():
    silences = [Silence(2.0, 2.4)]  # mid 2.2 < min_s
    spans = plan_chunks(40.0, silences, MAX, MIN)
    assert spans[0] == (0.0, 28.0)


def test_tail_never_shorter_than_min():
    # 30s with no silences: naive cut gives 28 + 2. Must instead split ~15/15.
    spans = plan_chunks(30.0, [], MAX, MIN)
    _assert_tiling(spans, 30.0)
    assert len(spans) == 2
    for s, e in spans:
        assert e - s >= MIN


def test_tail_split_uses_silence_when_available():
    spans = plan_chunks(31.0, [Silence(20.0, 20.5)], MAX, MIN)
    _assert_tiling(spans, 31.0)
    assert spans[0] == (0.0, 20.25)
    assert spans[1] == (20.25, 31.0)


@pytest.mark.parametrize("duration", [5.0, 27.9, 28.0, 28.1, 33.0, 56.0, 61.0, 5400.0])
def test_invariants_hold_for_many_durations(duration):
    silences = [Silence(t, t + 0.5) for t in range(7, int(duration), 13)]
    spans = plan_chunks(duration, silences, MAX, MIN)
    _assert_tiling(spans, duration)
    if duration >= MIN:
        for s, e in spans:
            assert e - s >= MIN - 1e-6


@requires_ffmpeg
def test_detect_silences_finds_gap(tmp_path: Path):
    wav = make_wav(tmp_path / "a.wav", [("tone", 2.0), ("silence", 1.0), ("tone", 2.0)])
    sil = detect_silences(wav)
    assert len(sil) == 1
    assert abs(sil[0].start - 2.0) < 0.1
    assert abs(sil[0].end - 3.0) < 0.1
    assert abs(sil[0].mid - 2.5) < 0.1


@requires_ffmpeg
def test_cut_chunks_writes_files_with_right_lengths(tmp_path: Path):
    wav = make_wav(tmp_path / "a.wav", [("tone", 10.0)])
    chunks = cut_chunks(wav, [(0.0, 4.0), (4.0, 10.0)], tmp_path / "chunks")
    assert [c.idx for c in chunks] == [0, 1]
    assert chunks[0].wav_path.name == "0000.wav"
    assert chunks[1].wav_path.exists()
    import wave
    with wave.open(str(chunks[1].wav_path), "rb") as w:
        assert abs(w.getnframes() / w.getframerate() - 6.0) < 0.05


@requires_ffmpeg
def test_chunk_audio_end_to_end(tmp_path: Path):
    parts = [("tone", 20.0), ("silence", 1.0), ("tone", 20.0), ("silence", 1.0), ("tone", 20.0)]
    wav = make_wav(tmp_path / "a.wav", parts)
    chunks = chunk_audio(wav, tmp_path / "chunks", MAX, MIN)
    assert len(chunks) == 3
    assert abs(chunks[0].end_s - 20.5) < 0.2
    assert all(c.duration_s <= MAX for c in chunks)
