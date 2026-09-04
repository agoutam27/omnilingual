"""Dataclasses passed between pipeline stages."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SegmentStatus = Literal["ok", "no_speech", "stt_failed", "mt_unsupported", "mt_failed"]


@dataclass(frozen=True)
class Chunk:
    idx: int
    start_s: float
    end_s: float
    wav_path: Path

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class STTResult:
    lang: str
    prob: float
    text: str


@dataclass
class Segment:
    chunk: Chunk
    lang: str
    prob: float
    text: str
    english: str | None
    status: SegmentStatus


@dataclass
class Cost:
    audio_seconds: float = 0.0
    mt_chars: int = 0
    inr_estimate: float = 0.0


@dataclass
class Transcript:
    source: Path
    duration_s: float
    segments: list[Segment] = field(default_factory=list)
    cost: Cost = field(default_factory=Cost)


def chunks_to_json(chunks: list[Chunk]) -> str:
    return json.dumps(
        [
            {"idx": c.idx, "start_s": c.start_s, "end_s": c.end_s, "wav_path": str(c.wav_path)}
            for c in chunks
        ],
        indent=2,
    )


def chunks_from_json(text: str) -> list[Chunk]:
    return [
        Chunk(idx=d["idx"], start_s=d["start_s"], end_s=d["end_s"], wav_path=Path(d["wav_path"]))
        for d in json.loads(text)
    ]
