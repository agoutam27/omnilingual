"""Runtime settings for omnilingual."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace


class ConfigError(Exception):
    """Raised when required configuration is missing."""


STT_PROVIDERS: tuple[str, ...] = ("sarvam", "mlx-whisper", "groq", "faster-whisper")

# stt_model=None means "the provider's own default", so flipping --stt does not
# drag a Sarvam model id into a Whisper provider (or vice versa).
DEFAULT_STT_MODELS: dict[str, str] = {
    "sarvam": "saaras:v4",
    "mlx-whisper": "mlx-community/whisper-large-v3-turbo",
    "groq": "whisper-large-v3-turbo",
    "faster-whisper": "small",
}

MT_PROVIDERS: tuple[str, ...] = ("mayura", "gemini", "indictrans2")

# Same rule as DEFAULT_STT_MODELS: mt_model=None means "the provider's own
# default", so --mt gemini never sends a Sarvam model id to Google.
DEFAULT_MT_MODELS: dict[str, str] = {
    "mayura": "mayura:v1",
    "gemini": "gemini-3.5-flash",
    # A community CTranslate2 conversion, not an ai4bharat release: converting the
    # official checkpoint needs torch, and the official repo is gated.
    "indictrans2": "adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M",
}

# Per-model input caps, in characters. Mayura rejects long inputs, so it is the
# tightest. Gemini's context window is orders of magnitude larger, so bigger
# pieces mean fewer requests and more context to resolve pronouns within a chunk.
# IndicTrans2 gets a small cap despite a far larger window: its own inference
# engine documents a 256-token limit and translates sentence-at-a-time, past
# which it repeats itself instead of translating.
MT_CHAR_LIMITS: dict[str, int] = {
    "mayura:v1": 1000,
    "sarvam-translate:v1": 2000,
    "adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M": 200,
}
DEFAULT_MT_CHAR_LIMIT = 4000


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str = "https://api.sarvam.ai"
    stt_provider: str = "sarvam"
    stt_model: str | None = None
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    diarizer: str | None = None
    num_speakers: int | None = None
    mt_provider: str = "mayura"
    mt_model: str | None = None
    mt_mode: str = "formal"
    max_chunk_s: float = 28.0
    min_chunk_s: float = 5.0
    langs: tuple[str, ...] = ()
    stt_inr_per_hour: float = 30.0
    mt_inr_per_10k_chars: float = 20.0

    @property
    def mt_char_limit(self) -> int:
        return MT_CHAR_LIMITS.get(self.resolved_mt_model, DEFAULT_MT_CHAR_LIMIT)

    @property
    def resolved_mt_model(self) -> str:
        try:
            default = DEFAULT_MT_MODELS[self.mt_provider]
        except KeyError:
            raise ConfigError(
                f"unknown MT provider {self.mt_provider!r} "
                f"(choose from: {', '.join(MT_PROVIDERS)})"
            ) from None
        return self.mt_model or default

    @property
    def resolved_stt_model(self) -> str:
        try:
            default = DEFAULT_STT_MODELS[self.stt_provider]
        except KeyError:
            raise ConfigError(
                f"unknown STT provider {self.stt_provider!r} "
                f"(choose from: {', '.join(STT_PROVIDERS)})"
            ) from None
        return self.stt_model or default

    def require_key(self) -> str:
        if not self.api_key:
            raise ConfigError(
                "Sarvam API key missing. Set SARVAM_API_KEY or pass --api-key."
            )
        return self.api_key

    def require_groq_key(self) -> str:
        if not self.groq_api_key:
            raise ConfigError(
                "Groq API key missing. Set GROQ_API_KEY."
            )
        return self.groq_api_key

    def require_gemini_key(self) -> str:
        if not self.gemini_api_key:
            raise ConfigError(
                "Gemini API key missing. Set GEMINI_API_KEY."
            )
        return self.gemini_api_key


def load_settings(
    api_key: str | None = None,
    env: Mapping[str, str] | None = None,
    **overrides,
) -> Settings:
    env = os.environ if env is None else env
    key = api_key or env.get("SARVAM_API_KEY") or None
    settings = Settings(
        api_key=key,
        groq_api_key=env.get("GROQ_API_KEY") or None,
        gemini_api_key=env.get("GEMINI_API_KEY") or None,
    )
    if "langs" in overrides:
        overrides["langs"] = tuple(overrides["langs"])
    settings = replace(settings, **overrides)
    # Validate both providers early, before any paid work.
    settings.resolved_stt_model
    settings.resolved_mt_model
    if settings.num_speakers is not None and settings.num_speakers < 2:
        raise ConfigError(
            f"num_speakers must be at least 2, got {settings.num_speakers}"
        )
    return settings
