"""Local Whisper speech-to-text on Intel/Linux via the optional faster-whisper extra."""

from __future__ import annotations

import importlib.util
import threading
from collections.abc import Callable
from pathlib import Path

from omnilingual.config import ConfigError, Settings
from omnilingual.models import STTResult
from omnilingual.stt.langs import to_bcp47

DecodeFn = Callable[[Path], "tuple[str, str, float]"]


class FasterWhisperSTT:
    """Whisper running on-device via CTranslate2: CPU-friendly, works on Intel Macs.

    decode is injectable so unit tests never touch the model weights.
    """

    mode = "transcribe"
    inr_per_hour = 0.0

    def __init__(self, settings: Settings, *, decode: DecodeFn | None = None) -> None:
        if decode is None:
            if importlib.util.find_spec("faster_whisper") is None:
                raise ConfigError(
                    "faster-whisper STT needs the local extra: "
                    "uv sync --extra local-stt  # or: uv pip install 'omnilingual[local-stt]'"
                )
            decode = _FasterEngine(settings.resolved_stt_model)
        self._decode = decode
        self.model = f"faster-whisper:{settings.resolved_stt_model}"
        # Live mode's stt_workers threads share this provider; CTranslate2
        # inference is not guaranteed thread-safe for a single model instance.
        self._lock = threading.Lock()

    def transcribe(self, wav_path: Path) -> STTResult:
        with self._lock:
            text, lang, prob = self._decode(wav_path)
        return STTResult(lang=to_bcp47(lang), prob=prob, text=text.strip())


class _FasterEngine:
    """Owns the WhisperModel and runs language detection + transcription per chunk."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            # CPU + int8 keeps memory low and stays fast on Intel i9 / Linux boxes;
            # faster-whisper will auto-use available CPU threads.
            self._model = WhisperModel(self._model_id, device="cpu", compute_type="int8")
        return self._model

    def __call__(self, wav_path: Path) -> tuple[str, str, float]:
        model = self._ensure_model()
        # vad_filter off — pipeline already did chunking on silence; language
        # auto-detect (language=None) gives info.language + info.language_probability.
        segments, info = model.transcribe(str(wav_path), beam_size=5)
        text = " ".join(seg.text for seg in segments).strip()
        lang = getattr(info, "language", None) or "en"
        prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        return text, lang, prob
