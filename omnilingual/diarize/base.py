"""Protocol every speaker-diarization backend satisfies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Turn:
    start_s: float
    end_s: float
    speaker: str  # raw engine label, e.g. "0", "1"; renumbered downstream


class Diarizer(Protocol):
    model: str

    def diarize(self, wav_path: Path) -> list[Turn]: ...
