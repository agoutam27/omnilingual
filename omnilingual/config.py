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


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str = "https://api.sarvam.ai"
    stt_provider: str = "sarvam"
    stt_model: str | None = None
    groq_api_key: str | None = None
    mt_model: str = "mayura:v1"
    mt_mode: str = "formal"
    max_chunk_s: float = 28.0
    min_chunk_s: float = 5.0
    langs: tuple[str, ...] = ()
    stt_inr_per_hour: float = 30.0
    mt_inr_per_10k_chars: float = 20.0

    @property
    def mt_char_limit(self) -> int:
        return 1000 if self.mt_model == "mayura:v1" else 2000

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


def load_settings(
    api_key: str | None = None,
    env: Mapping[str, str] | None = None,
    **overrides,
) -> Settings:
    env = os.environ if env is None else env
    key = api_key or env.get("SARVAM_API_KEY") or None
    settings = Settings(api_key=key, groq_api_key=env.get("GROQ_API_KEY") or None)
    if "langs" in overrides:
        overrides["langs"] = tuple(overrides["langs"])
    settings = replace(settings, **overrides)
    settings.resolved_stt_model  # validate stt_provider early, before any paid work
    return settings
