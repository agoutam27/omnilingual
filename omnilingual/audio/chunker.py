"""Split a normalized WAV into <=max_s chunks, cutting at silences when possible."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from omnilingual.audio.normalize import probe_duration, run_tool
from omnilingual.models import Chunk

_START = re.compile(r"silence_start:\s*([0-9.]+)")
_END = re.compile(r"silence_end:\s*([0-9.]+)")


@dataclass(frozen=True)
class Silence:
    start: float
    end: float

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2


def detect_silences(wav: Path, noise_db: float = -35.0, min_dur: float = 0.4) -> list[Silence]:
    # silencedetect writes its findings to stderr and ffmpeg exits 0 regardless,
    # so this intentionally does not use run_tool.
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats",
            "-i", str(wav),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    silences: list[Silence] = []
    start: float | None = None
    for line in proc.stderr.splitlines():
        if m := _START.search(line):
            start = float(m.group(1))
        elif (m := _END.search(line)) and start is not None:
            silences.append(Silence(start, float(m.group(1))))
            start = None
    if start is not None:  # silence ran to end of file
        silences.append(Silence(start, probe_duration(wav)))
    return silences


def _latest_mid_in(silences: list[Silence], lo: float, hi: float) -> float | None:
    mids = [s.mid for s in silences if lo <= s.mid <= hi]
    return max(mids) if mids else None


def _nearest_mid_to(silences: list[Silence], target: float, lo: float, hi: float) -> float | None:
    mids = [s.mid for s in silences if lo <= s.mid <= hi]
    return min(mids, key=lambda m: abs(m - target)) if mids else None


def plan_chunks(
    duration_s: float, silences: list[Silence], max_s: float, min_s: float
) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    while True:
        remaining = duration_s - cursor
        if remaining <= max_s:
            spans.append((cursor, duration_s))
            return spans
        if remaining < max_s + min_s:
            # A cut at max_s would leave a tail < min_s. Split the remainder in two.
            half = cursor + remaining / 2
            cut = _nearest_mid_to(silences, half, cursor + min_s, duration_s - min_s) or half
        else:
            cut = _latest_mid_in(silences, cursor + min_s, cursor + max_s) or (cursor + max_s)
        spans.append((cursor, cut))
        cursor = cut


def cut_chunks(wav: Path, spans: list[tuple[float, float]], out_dir: Path) -> list[Chunk]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []
    for idx, (start, end) in enumerate(spans):
        dst = out_dir / f"{idx:04d}.wav"
        run_tool(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(wav),
                "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                "-c:a", "pcm_s16le",
                str(dst),
            ],
        )
        chunks.append(Chunk(idx=idx, start_s=start, end_s=end, wav_path=dst))
    return chunks


def chunk_audio(wav: Path, out_dir: Path, max_s: float, min_s: float) -> list[Chunk]:
    duration = probe_duration(wav)
    spans = plan_chunks(duration, detect_silences(wav), max_s, min_s)
    return cut_chunks(wav, spans, out_dir)
