"""Protocol every speech-to-text backend satisfies."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from omnilingual.models import STTResult


class STTProvider(Protocol):
    model: str
    mode: str

    def transcribe(self, wav_path: Path) -> STTResult: ...
