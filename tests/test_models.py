from pathlib import Path

from omnilingual.models import (
    Chunk,
    Cost,
    Segment,
    STTResult,
    Transcript,
    chunks_from_json,
    chunks_to_json,
)


def test_chunk_duration():
    c = Chunk(idx=0, start_s=1.5, end_s=4.0, wav_path=Path("/tmp/x.wav"))
    assert c.duration_s == 2.5


def test_chunks_json_roundtrip():
    chunks = [
        Chunk(0, 0.0, 27.5, Path("chunks/0000.wav")),
        Chunk(1, 27.5, 50.0, Path("chunks/0001.wav")),
    ]
    text = chunks_to_json(chunks)
    assert chunks_from_json(text) == chunks


def test_segment_defaults():
    c = Chunk(0, 0.0, 10.0, Path("a.wav"))
    seg = Segment(chunk=c, lang="hi-IN", prob=0.9, text="नमस्ते", english="Hello", status="ok")
    assert seg.english == "Hello"
    assert seg.status == "ok"


def test_transcript_holds_cost():
    t = Transcript(source=Path("m.m4a"), duration_s=60.0, segments=[], cost=Cost())
    assert t.cost.inr_estimate == 0.0
    assert STTResult("ta-IN", 0.8, "வணக்கம்").lang == "ta-IN"
