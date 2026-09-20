from pathlib import Path

from omnilingual.diarize.assign import assign_speakers, renumber
from omnilingual.diarize.base import Turn
from omnilingual.models import Chunk


def _chunk(idx: int, start: float, end: float) -> Chunk:
    return Chunk(idx=idx, start_s=start, end_s=end, wav_path=Path(f"{idx:04d}.wav"))


def test_dominant_overlap_wins():
    chunks = [_chunk(0, 0.0, 10.0)]
    turns = [Turn(0.0, 8.0, "1"), Turn(8.0, 10.0, "0")]
    assert assign_speakers(chunks, turns) == {0: "Speaker 1"}


def test_exact_tie_resolves_to_previous_speaker():
    chunks = [_chunk(0, 0.0, 10.0), _chunk(1, 10.0, 20.0)]
    turns = [Turn(0.0, 10.0, "7"), Turn(10.0, 15.0, "7"), Turn(15.0, 20.0, "3")]
    assert assign_speakers(chunks, turns) == {0: "Speaker 1", 1: "Speaker 1"}


def test_chunk_with_no_overlap_gets_none():
    chunks = [_chunk(0, 0.0, 10.0)]
    assert assign_speakers(chunks, [Turn(20.0, 30.0, "0")]) == {0: None}
    assert assign_speakers(chunks, []) == {0: None}


def test_larger_overlap_wins_among_overlapping_turns():
    chunks = [_chunk(0, 0.0, 5.0), _chunk(1, 5.0, 15.0)]
    turns = [Turn(0.0, 5.0, "0"), Turn(5.0, 8.0, "0"), Turn(7.0, 15.0, "1")]
    # chunk 1: 3 s overlap with "0" vs 8 s with "1" → "1", seen second → Speaker 2
    assert assign_speakers(chunks, turns) == {0: "Speaker 1", 1: "Speaker 2"}


def test_renumber_follows_first_appearance():
    assert renumber(["4", "2", "4", "2", "9"]) == [
        "Speaker 1",
        "Speaker 2",
        "Speaker 1",
        "Speaker 2",
        "Speaker 3",
    ]
    assert renumber([None, "0"]) == [None, "Speaker 1"]
