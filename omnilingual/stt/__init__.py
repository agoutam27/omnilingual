"""STT provider factory: the single place that turns Settings into a provider."""

from __future__ import annotations

from omnilingual.config import Settings
from omnilingual.stt.base import STTProvider


def build_stt(settings: Settings) -> STTProvider:
    if settings.stt_provider == "mlx-whisper":
        from omnilingual.stt.mlx_whisper import MlxWhisperSTT

        return MlxWhisperSTT(settings)
    if settings.stt_provider == "groq":
        from omnilingual.stt.groq import GroqSTT

        return GroqSTT(settings)
    if settings.stt_provider == "faster-whisper":
        from omnilingual.stt.faster_whisper import FasterWhisperSTT

        return FasterWhisperSTT(settings)

    from omnilingual.stt.sarvam import SarvamSTT

    return SarvamSTT(settings)
