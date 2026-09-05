"""Crash-safe incremental Markdown writer for live sessions.

The 4-line header block is exactly HEADER_WIDTH bytes, space-padded, and
rewritten in place with pwrite so the inode never changes (tail -f safe).
Segment bodies are single O_APPEND writes followed by fsync.
"""

from __future__ import annotations

import os
from pathlib import Path

from omnilingual.models import Segment

from .markdown import (
    format_english_line,
    format_segment,
    fmt_ts,
    lang_share,
)

HEADER_WIDTH = 512
_LINES = 4
_LINE_WIDTH = HEADER_WIDTH // _LINES  # 128 bytes per line incl. newline


def _pad(line: str) -> bytes:
    raw = (line + "\n").encode("utf-8")
    if len(raw) > _LINE_WIDTH:
        raw = (line[: _LINE_WIDTH - 5] + "…" + "\n").encode("utf-8")
    return raw + b" " * (_LINE_WIDTH - len(raw))


class LiveMarkdownWriter:
    """Appends segments to a growing Markdown file, header kept current."""

    heading = "## Transcript"

    def __init__(self, path: Path, *, title: str, cost_cap: float) -> None:
        self.fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        self._fd = self.fd
        self.title = title
        self.cost_cap = cost_cap
        self._segments: list[Segment] = []
        self._cost = 0.0
        self._growing = True
        os.write(self._fd, self._header())
        static = f"{self.heading}\n\n".encode("utf-8")
        os.write(self._fd, static)
        os.fsync(self._fd)

    def _header(self) -> bytes:
        n = len(self._segments)
        shares = ", ".join(
            f"{lang} {pct}%" for lang, pct in lang_share(self._segments)
        ) or "—"
        # Duration so far = end of the last appended segment.
        dur = self._segments[-1].chunk.end_s if self._segments else 0.0
        growing = " · growing" if self._growing else ""
        return b"".join(
            [
                _pad(f"# {self.title}"),
                _pad(f"Duration (so far) {fmt_ts(dur)} · {n} segments · {shares}"),
                _pad(f"Cost ₹{self._cost:.2f} (cap ₹{self.cost_cap:.0f}){growing}"),
                _pad(""),
            ]
        )

    def append_segment(self, seg: Segment, cost_delta: float = 0.0) -> None:
        # Batch-identical layout: format_segment's trailing "" is the
        # inter-segment separator, so drop it here and re-add it as the
        # leading "\n" of every segment after the first.
        block = format_segment(seg)[:-1]
        body = ((("\n" if self._segments else "") + "\n".join(block) + "\n")
                .encode("utf-8"))
        os.write(self._fd, body)  # O_APPEND single write
        self._segments.append(seg)
        self._cost += cost_delta
        os.pwrite(self._fd, self._header(), 0)
        os.fsync(self._fd)

    def finalize(self) -> None:
        self._growing = False
        os.pwrite(self._fd, self._header(), 0)
        os.fsync(self._fd)

    def close(self) -> None:
        fd, self._fd = self._fd, -1
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> LiveMarkdownWriter:
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.finalize()
        finally:
            self.close()


class LiveEnglishWriter(LiveMarkdownWriter):
    """Mirrors the English-only file; no_speech segments are a no-op."""

    heading = "## Transcript (English)"

    def append_segment(self, seg: Segment, cost_delta: float = 0.0) -> None:
        line = format_english_line(seg)
        if line is None:
            return
        body = ((("\n" if self._segments else "") + line + "\n")
                .encode("utf-8"))
        os.write(self._fd, body)
        self._segments.append(seg)
        self._cost += cost_delta
        os.pwrite(self._fd, self._header(), 0)
        os.fsync(self._fd)
