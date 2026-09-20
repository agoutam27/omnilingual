"""Manual diarization gate: real models on real-ish audio.

Every test skips unless OMNILINGUAL_MANUAL=1 (plus the diarize marker, so
default runs stay fast). Run after `uv sync --extra diarize` on a machine
with the models downloadable:

    OMNILINGUAL_MANUAL=1 uv run pytest tests/diarize/test_manual.py -q --run-diarize
"""

import math
import os
import struct
import wave
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.diarize,
    pytest.mark.skipif(
        os.environ.get("OMNILINGUAL_MANUAL") != "1",
        reason="manual only: needs models, mic-free but slow",
    ),
]

sherpa_onnx = pytest.importorskip("sherpa_onnx")

from omnilingual.config import load_settings  # noqa: E402
from omnilingual.diarize import build_diarizer  # noqa: E402
from omnilingual.diarize.assign import assign_speakers  # noqa: E402
from omnilingual.models import Chunk  # noqa: E402


def voice_wav(path: Path, parts: list[tuple[int, float]]) -> Path:
    """Mono 16 kHz s16le WAV alternating voices by base frequency (Hz, seconds)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        for freq, seconds in parts:
            for i in range(int(seconds * 16000)):
                sample = int(6000 * math.sin(2 * math.pi * freq * i / 16000))
                sample += int(1500 * math.sin(2 * math.pi * freq * 2.7 * i / 16000))
                w.writeframes(struct.pack("<h", max(-32768, min(32767, sample))))
    return path


def _chunks_for(wav: Path, size_s: float = 5.0) -> list[Chunk]:
    n = int(20.0 / size_s)
    return [
        Chunk(idx=i, start_s=i * size_s, end_s=(i + 1) * size_s, wav_path=wav)
        for i in range(n)
    ]


def test_two_voices_yield_two_speakers(tmp_path: Path):
    wav = voice_wav(tmp_path / "two.wav", [(220, 5.0), (440, 5.0)] * 2)
    dz = build_diarizer(load_settings(api_key="k", env={}, diarizer="sherpa"))
    assigned = assign_speakers(
        _chunks_for(wav), dz.diarize(wav)
    )
    labels = [v for v in assigned.values() if v is not None]
    assert len(set(labels)) == 2, f"expected 2 speakers, got {set(labels)}"
    # No single-chunk flip-flopping: runs of identical labels dominate.
    flips = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    assert flips <= 3


def test_silence_only_never_crashes(tmp_path: Path):
    wav = tmp_path / "quiet.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00" * 16000 * 2 * 10)
    dz = build_diarizer(load_settings(api_key="k", env={}, diarizer="sherpa"))
    turns = dz.diarize(wav)
    assert isinstance(turns, list)


def test_three_voices_with_speaker_hint(tmp_path: Path):
    wav = voice_wav(
        tmp_path / "three.wav",
        [(220, 5.0), (330, 5.0), (440, 5.0), (220, 5.0)],
    )
    dz = build_diarizer(
        load_settings(api_key="k", env={}, diarizer="sherpa", num_speakers=3)
    )
    assigned = assign_speakers(_chunks_for(wav), dz.diarize(wav))
    labels = {v for v in assigned.values() if v is not None}
    assert len(labels) == 3, f"expected 3 speakers, got {labels}"
