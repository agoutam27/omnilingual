from omnilingual.audio.live_slicer import LiveSlicer
from omnilingual.models import chunks_from_json
from tests.conftest import raw_pcm


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


def test_gap_audio_discarded_from_next_chunk(tmp_path):
    session = tmp_path / "s"
    s = LiveSlicer(session, target_s=8.0, max_s=28.0, min_s=5.0)
    s.feed(raw_pcm([("tone", 10.0)]))
    s.note_gap(10.0, 11.0)
    sealed = s.feed(raw_pcm([("tone", 8.0)]))
    assert sealed == []
    sealed = s.note_gap(19.0, 19.5)
    assert [(c.chunk.start_s, c.chunk.end_s) for c in sealed] == [(11.0, 19.0)]


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
