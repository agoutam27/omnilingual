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


def raw_pcm(parts: list[tuple[str, float]], rate: int = 16000) -> bytes:
    """Mono s16le PCM for ("tone"|"silence", seconds) parts. Test helper."""
    out = bytearray()
    for kind, seconds in parts:
        n = int(seconds * rate)
        if kind == "tone":
            for i in range(n):
                out += struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
        elif kind == "silence":
            out += b"\x00" * (n * 2)
        else:
            raise ValueError(f"unknown part kind: {kind}")
    return bytes(out)


def make_wav(
    path: Path,
    parts: list[tuple[str, float]],
    rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Write a 16-bit PCM WAV built from ("tone"|"silence", seconds) parts."""
    data = raw_pcm(parts, rate)
    if channels == 2:
        interleaved = bytearray()
        for i in range(0, len(data), 2):
            interleaved += data[i : i + 2] * 2
        data = bytes(interleaved)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data)
    return path
