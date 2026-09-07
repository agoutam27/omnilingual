import pytest

from omnilingual.audio.live_slicer import LiveSlicer
from omnilingual.models import chunks_from_json
from tests.conftest import raw_pcm


def test_fractional_gap_start_never_yields_odd_head(tmp_path):
    """ffmpeg gap timestamps are fractional seconds; int() truncation of the
    byte length must not produce an odd s16le count (crashed the capture
    thread in the field: 10.000046875 * 32000 truncates to 320001 bytes)."""
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=8.0, max_s=28.0, min_s=5.0)
    s.feed(raw_pcm([("tone", 11.0)]))
    sealed = s.note_gap(10.000046875, 10.8)
    assert [(c.chunk.start_s, c.chunk.end_s) for c in sealed] == [(0.0, 10.000046875)]
    assert sealed[0].speech is True


def test_seals_at_first_gap_after_target(tmp_path):
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=8.0, max_s=28.0, min_s=5.0)
    sealed = s.feed(raw_pcm([("tone", 10.0)]))
    assert sealed == []  # gap not known yet: nothing sealed
    sealed = s.note_gap(10.0, 10.8)
    assert [c.chunk.start_s for c in sealed] == [0.0]
    assert sealed[0].chunk.end_s == 10.0
    assert sealed[0].speech is True


def test_ignores_gap_before_target(tmp_path):
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=8.0, max_s=28.0, min_s=5.0)
    s.feed(raw_pcm([("tone", 4.0)]))
    assert s.note_gap(4.0, 4.5) == []
    sealed = s.feed(raw_pcm([("tone", 6.0)]))
    assert sealed == []
    sealed = s.note_gap(10.0, 10.6)
    assert [(c.chunk.start_s, c.chunk.end_s) for c in sealed] == [(0.0, 10.0)]


def test_gap_preroll_capped_at_preroll_s(tmp_path):
    """A 1 s gap keeps only its final preroll_s (0.75 s) as the next chunk's
    pre-roll; the excess 0.25 s (8000 bytes) is dropped byte-exactly, so the
    next chunk starts at 11.0 - 0.75 = 10.25."""
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=8.0, max_s=28.0, min_s=5.0)
    s.feed(raw_pcm([("tone", 10.0)]))
    s.feed(raw_pcm([("silence", 1.0)]))  # gap audio streams through the buffer
    s.note_gap(10.0, 11.0)
    sealed = s.feed(raw_pcm([("tone", 8.0)]))
    assert sealed == []
    s.feed(raw_pcm([("silence", 0.5)]))
    sealed = s.note_gap(19.0, 19.5)
    assert [(c.chunk.start_s, c.chunk.end_s) for c in sealed] == [(10.25, 19.0)]


def test_short_gap_kept_entirely_next_chunk_contiguous(tmp_path):
    """A gap shorter than preroll_s is kept whole: the next chunk starts
    exactly where the previous one ended, so no audio is lost or doubled."""
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=8.0, max_s=28.0, min_s=5.0)
    s.feed(raw_pcm([("tone", 10.0)]))
    s.feed(raw_pcm([("silence", 0.6)]))
    assert [(c.chunk.start_s, c.chunk.end_s) for c in s.note_gap(10.0, 10.6)] == [(0.0, 10.0)]
    s.feed(raw_pcm([("tone", 8.4)]))  # 10.6 -> 19.0
    s.feed(raw_pcm([("silence", 0.5)]))  # 19.0 -> 19.5
    sealed = s.note_gap(19.0, 19.5)
    assert [(c.chunk.start_s, c.chunk.end_s) for c in sealed] == [(10.0, 19.0)]


def test_long_gap_preroll_starts_at_gap_end_minus_preroll(tmp_path):
    """A 5 s silence keeps only its final 0.75 s: the next chunk starts at
    gap_end - preroll_s (15.0 - 0.75 = 14.25, an exact 136000-byte drop), so
    rms speech gating is not diluted by minutes of silence."""
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=8.0, max_s=28.0, min_s=5.0)
    s.feed(raw_pcm([("tone", 10.0)]))
    s.feed(raw_pcm([("silence", 5.0)]))
    assert [(c.chunk.start_s, c.chunk.end_s) for c in s.note_gap(10.0, 15.0)] == [(0.0, 10.0)]
    s.feed(raw_pcm([("tone", 8.0)]))  # 15.0 -> 23.0
    s.feed(raw_pcm([("silence", 0.5)]))  # 23.0 -> 23.5
    sealed = s.note_gap(23.0, 23.5)
    assert [(c.chunk.start_s, c.chunk.end_s) for c in sealed] == [(14.25, 23.0)]
    manifest = chunks_from_json((session / "chunks.json").read_text())
    assert [(c.idx, c.start_s, c.end_s) for c in manifest] == [(0, 0.0, 10.0), (1, 14.25, 23.0)]


def test_zero_preroll_restores_full_gap_drop(tmp_path):
    """preroll_s=0 keeps none of the gap: the old full-drop behavior."""
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=8.0, max_s=28.0, min_s=5.0, preroll_s=0.0)
    s.feed(raw_pcm([("tone", 10.0)]))
    s.feed(raw_pcm([("silence", 1.0)]))
    s.note_gap(10.0, 11.0)
    sealed = s.feed(raw_pcm([("tone", 8.0)]))
    assert sealed == []
    s.feed(raw_pcm([("silence", 0.5)]))
    sealed = s.note_gap(19.0, 19.5)
    assert [(c.chunk.start_s, c.chunk.end_s) for c in sealed] == [(11.0, 19.0)]


def test_stale_gap_pruned_after_preroll_seal(tmp_path):
    """The processed gap must not linger over the kept pre-roll region (the
    new chunk start sits inside the old gap), and a stale gap inside the
    pre-roll must never re-seal; the next real gap seals with exact bounds."""
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=8.0, max_s=28.0, min_s=5.0)
    s.feed(raw_pcm([("tone", 10.0)]))
    s.feed(raw_pcm([("silence", 0.8)]))
    assert [(c.chunk.start_s, c.chunk.end_s) for c in s.note_gap(10.0, 10.8)] == [(0.0, 10.0)]
    assert s._gaps == []  # processed gap pruned although the new start (10.05) is inside it
    s.note_gap(10.2, 10.5)  # stale: entirely inside the kept pre-roll region
    s.feed(raw_pcm([("tone", 8.2)]))  # 10.8 -> 19.0
    assert s.feed(raw_pcm([("silence", 0.5)])) == []  # 19.0 -> 19.5; stale gap never seals
    sealed = s.note_gap(19.0, 19.5)
    assert [(c.chunk.start_s, c.chunk.end_s) for c in sealed] == [(10.05, 19.0)]
    assert s._gaps == []


def test_rejects_negative_preroll(tmp_path):
    with pytest.raises(ValueError, match="preroll_s"):
        LiveSlicer(tmp_path / "s", preroll_s=-0.1)


def test_hard_cut_at_max_without_silence(tmp_path):
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=8.0, max_s=12.0, min_s=5.0)
    sealed = s.feed(raw_pcm([("tone", 12.0)]))
    assert [(c.chunk.start_s, c.chunk.end_s) for c in sealed] == [(0.0, 12.0)]


def test_flush_seals_partial_above_min_and_discards_below(tmp_path):
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=8.0, max_s=28.0, min_s=5.0)
    s.feed(raw_pcm([("tone", 6.0)]))
    sealed = s.flush()
    assert [(c.chunk.start_s, c.chunk.end_s) for c in sealed] == [(0.0, 6.0)]
    s.feed(raw_pcm([("tone", 2.0)]))
    assert s.flush() == []


def test_quiet_chunk_tagged_no_speech_and_excluded_from_manifest(tmp_path):
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=4.0, max_s=6.0, min_s=2.0)
    sealed = s.feed(raw_pcm([("silence", 6.0)]))
    assert len(sealed) == 1 and sealed[0].speech is False
    manifest = chunks_from_json((session / "chunks.json").read_text())
    assert manifest == []
    assert not (session / "live-chunks" / "0000.wav").exists()


def test_manifest_round_trips_speech_chunks(tmp_path):
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=4.0, max_s=6.0, min_s=2.0)
    s.feed(raw_pcm([("tone", 6.0)]))
    manifest = chunks_from_json((session / "chunks.json").read_text())
    assert [(c.idx, c.start_s, c.end_s) for c in manifest] == [(0, 0.0, 6.0)]
    import wave

    with wave.open(str(manifest[0].wav_path), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert w.readframes(w.getnframes()) == raw_pcm([("tone", 6.0)])
