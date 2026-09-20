from pathlib import Path

from omnilingual.models import Chunk, Cost, Segment, Transcript
from omnilingual.render.markdown import (
    fmt_ts,
    lang_share,
    render,
    render_english_only,
    speaker_share,
)

GOLDEN = Path(__file__).parent / "golden"


def _chunk(i: int) -> Chunk:
    return Chunk(idx=i, start_s=i * 25.0, end_s=(i + 1) * 25.0, wav_path=Path(f"{i:04d}.wav"))


def sample_transcript() -> Transcript:
    segs = [
        Segment(_chunk(0), "hi-IN", 0.97, "हम आज payment dashboard के बारे में बात करेंगे।",
                "We will talk about the payment dashboard today.", "ok"),
        Segment(_chunk(1), "en-IN", 0.99, "Okay, let's start with the refund numbers.", None, "ok"),
        Segment(_chunk(2), "ta-IN", 0.91, "வணக்கம் எல்லோருக்கும்.", None, "mt_unsupported"),
        Segment(_chunk(3), "hi-IN", 0.0, "[transcription failed]", None, "stt_failed"),
    ]
    return Transcript(source=Path("/rec/standup.m4a"), duration_s=100.0, segments=segs,
                      cost=Cost(audio_seconds=100.0, mt_chars=47, inr_estimate=1.234))


def test_fmt_ts():
    assert fmt_ts(0) == "00:00:00"
    assert fmt_ts(59.9) == "00:00:59"
    assert fmt_ts(3661) == "01:01:01"


def test_lang_share_by_duration_desc():
    assert lang_share(sample_transcript().segments) == [("hi-IN", 50), ("en-IN", 25), ("ta-IN", 25)]


def test_lang_share_empty():
    assert lang_share([]) == []


def test_render_matches_golden():
    assert render(sample_transcript()) == (GOLDEN / "interleaved.md").read_text(encoding="utf-8")


def test_render_english_only_matches_golden():
    assert render_english_only(sample_transcript()) == (GOLDEN / "english_only.md").read_text(encoding="utf-8")


def test_mt_failed_note():
    t = sample_transcript()
    t.segments = [Segment(_chunk(0), "hi-IN", 0.9, "क", None, "mt_failed")]
    out = render(t)
    assert "_(translation failed)_" in out
    assert "> " not in out


def _no_speech_transcript() -> Transcript:
    segs = [
        Segment(_chunk(0), "hi-IN", 0.9, "क", "EN[क]", "ok"),
        Segment(_chunk(1), "hi-IN", 0.12, "", None, "no_speech"),
        Segment(_chunk(2), "en-IN", 0.99, "hello", None, "ok"),
    ]
    return Transcript(source=Path("/rec/standup.m4a"), duration_s=75.0, segments=segs,
                      cost=Cost(audio_seconds=75.0, mt_chars=1, inr_estimate=1.0))


def test_no_speech_renders_header_note_and_no_text_line():
    out = render(_no_speech_transcript())
    assert "**[00:00:25 → 00:00:50] hi-IN** _(no speech detected)_\n\n**[00:00:50" in out
    assert out.count("> ") == 1  # only the hi-IN ok segment carries a translation


def test_no_speech_excluded_from_lang_share():
    # chunk 1 is 25s of hi-IN-labelled silence; it must not inflate hi-IN's share
    assert lang_share(_no_speech_transcript().segments) == [("en-IN", 50), ("hi-IN", 50)]


def test_lang_share_of_only_no_speech_is_empty():
    segs = [Segment(_chunk(0), "hi-IN", 0.1, "", None, "no_speech")]
    assert lang_share(segs) == []


def test_english_only_skips_no_speech_segments():
    out = render_english_only(_no_speech_transcript())
    assert out == "# Meeting transcript — standup.m4a (English)\n\nEN[क]\n\nhello\n"


def test_ok_segment_without_english_has_no_quote_line():
    t = _no_speech_transcript()
    t.segments = [Segment(_chunk(0), "hi-IN", 0.9, "क", None, "ok")]
    out = render(t)
    assert "क" in out
    assert "> " not in out


from omnilingual.render.markdown import format_english_line, format_segment


def test_format_segment_ok_with_translation():
    seg = Segment(_chunk(0), "hi-IN", 0.97, "नमस्ते", "Hello", "ok")
    assert format_segment(seg) == [
        "**[00:00:00 → 00:00:25] hi-IN**",
        "नमस्ते",
        "> Hello",
        "",
    ]


def test_format_segment_ok_english_has_no_quote_line():
    seg = Segment(_chunk(0), "en-IN", 0.99, "hello", None, "ok")
    assert format_segment(seg) == ["**[00:00:00 → 00:00:25] en-IN**", "hello", ""]


def test_format_segment_no_speech_has_note_and_no_text():
    seg = Segment(_chunk(1), "hi-IN", 0.1, "", None, "no_speech")
    assert format_segment(seg) == ["**[00:00:25 → 00:00:50] hi-IN** _(no speech detected)_", ""]


def test_format_english_line_variants():
    assert format_english_line(Segment(_chunk(0), "hi-IN", 0.9, "क", "EN[क]", "ok")) == "EN[क]"
    assert format_english_line(Segment(_chunk(0), "en-IN", 0.9, "hello", None, "ok")) == "hello"
    assert format_english_line(Segment(_chunk(0), "hi-IN", 0.0, "[transcription failed]", None, "stt_failed")) == "[transcription failed]"
    assert format_english_line(Segment(_chunk(0), "ta-IN", 0.9, "வ", None, "mt_failed")) == "[ta-IN, untranslated]"
    assert format_english_line(Segment(_chunk(0), "hi-IN", 0.1, "", None, "no_speech")) is None


def test_format_segment_with_speaker_prefixes_head():
    seg = Segment(_chunk(0), "hi-IN", 0.97, "नमस्ते", "Hello", "ok", speaker="Speaker 2")
    assert format_segment(seg) == [
        "**[00:00:00 → 00:00:25] Speaker 2 · hi-IN**",
        "नमस्ते",
        "> Hello",
        "",
    ]


def test_format_segment_without_speaker_is_unchanged():
    seg = Segment(_chunk(0), "hi-IN", 0.97, "नमस्ते", "Hello", "ok")
    assert format_segment(seg)[0] == "**[00:00:00 → 00:00:25] hi-IN**"


def test_format_english_line_prefixes_speaker():
    assert format_english_line(Segment(_chunk(0), "hi-IN", 0.9, "क", "EN[क]", "ok", speaker="Speaker 2")) == "Speaker 2: EN[क]"
    assert format_english_line(Segment(_chunk(0), "en-IN", 0.9, "hello", None, "ok", speaker="Speaker 1")) == "Speaker 1: hello"
    assert format_english_line(Segment(_chunk(0), "hi-IN", 0.0, "[transcription failed]", None, "stt_failed", speaker="Speaker 2")) == "Speaker 2: [transcription failed]"
    assert format_english_line(Segment(_chunk(0), "ta-IN", 0.9, "வ", None, "mt_failed", speaker="Speaker 2")) == "Speaker 2: [ta-IN, untranslated]"
    assert format_english_line(Segment(_chunk(0), "hi-IN", 0.1, "", None, "no_speech", speaker="Speaker 2")) is None


def test_speaker_share_reports_speech_share_and_omits_silence():
    segs = [
        Segment(_chunk(0), "hi-IN", 0.9, "क", "EN", "ok", speaker="Speaker 1"),
        Segment(_chunk(1), "ta-IN", 0.9, "வ", None, "mt_failed", speaker="Speaker 2"),
        Segment(_chunk(2), "hi-IN", 0.1, "", None, "no_speech", speaker="Speaker 2"),
    ]
    assert speaker_share(segs) == [("Speaker 1", 50), ("Speaker 2", 50)]
    assert speaker_share([]) == []


def test_render_adds_speaker_line_only_when_labels_exist():
    labelled = sample_transcript()
    labelled.segments[0] = Segment(
        labelled.segments[0].chunk,
        labelled.segments[0].lang,
        labelled.segments[0].prob,
        labelled.segments[0].text,
        labelled.segments[0].english,
        labelled.segments[0].status,
        speaker="Speaker 1",
    )
    labelled.segments[1] = Segment(
        labelled.segments[1].chunk,
        labelled.segments[1].lang,
        labelled.segments[1].prob,
        labelled.segments[1].text,
        labelled.segments[1].english,
        labelled.segments[1].status,
        speaker="Speaker 2",
    )
    out = render(labelled)
    assert "**[00:00:00 → 00:00:25] Speaker 1 · hi-IN**" in out
    assert "Speakers: Speaker 1 50%, Speaker 2 50%" in out
    assert "Speakers:" not in render(sample_transcript())
