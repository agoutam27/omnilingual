import os

import pytest

from omnilingual.models import Chunk, Segment
from omnilingual.render.live import (
    HEADER_WIDTH,
    LiveEnglishWriter,
    LiveMarkdownWriter,
)

CHUNK = Chunk(idx=0, start_s=4.0, end_s=13.0, wav_path="x.wav")


def _seg(status="ok"):
    return Segment(chunk=CHUNK, lang="hi-IN", prob=0.97,
                   text="Namaste", english="Hello", status=status)


def test_header_block_is_fixed_width_and_updated_in_place(tmp_path):
    out = tmp_path / "meeting.md"
    with LiveMarkdownWriter(out, title="T", cost_cap=50.0) as w:
        ino = os.stat(out).st_ino
        assert os.stat(out).st_size >= HEADER_WIDTH
        w.append_segment(_seg(), cost_delta=0.05)
        assert os.stat(out).st_ino == ino
    text = out.read_text(encoding="utf-8")
    assert "00:00:04 → 00:00:13" in text
    assert "Namaste" in text and "> Hello" in text
    assert "· growing" not in text  # finalize() dropped the marker


def test_growing_marker_present_mid_session(tmp_path):
    out = tmp_path / "meeting.md"
    w = LiveMarkdownWriter(out, title="T", cost_cap=50.0)
    w.append_segment(_seg(), cost_delta=0.05)
    text = out.read_text(encoding="utf-8")
    assert "· growing" in text
    assert "Cost ₹0.05 (cap ₹50)" in text
    w.close()


def test_crash_mid_session_leaves_valid_file(tmp_path):
    out = tmp_path / "meeting.md"
    w = LiveMarkdownWriter(out, title="T", cost_cap=50.0)
    w.append_segment(_seg(), cost_delta=0.05)
    w.append_segment(_seg(status="stt_failed"), cost_delta=0.0)
    os.close(w._fd)  # simulate kill -9: no finalize, no close
    text = out.read_text(encoding="utf-8")  # must still parse as text
    assert text.count("**[00:00:04 → 00:00:13]") == 2
    assert "transcription failed" in text


def test_existing_file_is_never_truncated(tmp_path):
    out = tmp_path / "meeting.md"
    out.write_text("precious", encoding="utf-8")
    with pytest.raises(FileExistsError):
        LiveMarkdownWriter(out, title="T", cost_cap=50.0)
    assert out.read_text(encoding="utf-8") == "precious"


def test_english_writer_skips_no_speech(tmp_path):
    out = tmp_path / "meeting.en.md"
    with LiveEnglishWriter(out, title="T", cost_cap=50.0) as w:
        w.append_segment(_seg(status="no_speech"))
        w.append_segment(_seg())
    text = out.read_text(encoding="utf-8")
    assert text.count("Hello") == 1  # ok segment's English line, exactly once
    assert "Namaste" not in text  # no_speech segment contributed nothing
