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


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-live", action="store_true", default=False, help="run tests marked live (real Sarvam API)")
    parser.addoption("--run-mlx", action="store_true", default=False, help="run tests marked mlx (real mlx-whisper model)")
    parser.addoption(
        "--run-faster-whisper",
        action="store_true",
        default=False,
        help="run tests marked faster_whisper (real faster-whisper model, downloads ~500 MB)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # If the user explicitly filtered by marker (-m) or keyword (-k), let pytest
    # handle it — don't second-guess. Auto-skip only for the default `pytest` run.
    markexpr: str = config.getoption("markexpr") or ""
    keyword: str = config.getoption("keyword") or ""
    if markexpr or keyword:
        return
    skip_live = pytest.mark.skip(reason="need --run-live to run live tests")
    skip_mlx = pytest.mark.skip(reason="need --run-mlx to run mlx tests (Apple Silicon + local-stt extra)")
    skip_fw = pytest.mark.skip(reason="need --run-faster-whisper to run faster_whisper tests (downloads ~500 MB)")
    for item in items:
        if not config.getoption("--run-live") and "live" in item.keywords:
            item.add_marker(skip_live)
        if not config.getoption("--run-mlx") and "mlx" in item.keywords:
            item.add_marker(skip_mlx)
        if not config.getoption("--run-faster-whisper") and "faster_whisper" in item.keywords:
            item.add_marker(skip_fw)


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
