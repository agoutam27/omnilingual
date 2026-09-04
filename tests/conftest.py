import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def make_wav(
    path: Path,
    parts: list[tuple[str, float]],
    rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Write a 16-bit PCM WAV built from ("tone"|"silence", seconds) parts."""
    frames = bytearray()
    for kind, seconds in parts:
        n = int(seconds * rate)
        for i in range(n):
            if kind == "tone":
                sample = int(8000 * math.sin(2 * math.pi * 440 * i / rate))
            else:
                sample = 0
            frames += struct.pack("<h", sample) * channels
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return path
