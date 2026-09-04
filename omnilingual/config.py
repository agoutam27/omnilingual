"""Runtime settings for omnilingual."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace


class ConfigError(Exception):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str = "https://api.sarvam.ai"
    stt_model: str = "saaras:v4"
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

    def require_key(self) -> str:
        if not self.api_key:
            raise ConfigError(
                "Sarvam API key missing. Set SARVAM_API_KEY or pass --api-key."
            )
        return self.api_key


def load_settings(
    api_key: str | None = None,
    env: Mapping[str, str] | None = None,
    **overrides,
) -> Settings:
    env = os.environ if env is None else env
    key = api_key or env.get("SARVAM_API_KEY") or None
    settings = Settings(api_key=key)
    if "langs" in overrides:
        overrides["langs"] = tuple(overrides["langs"])
    return replace(settings, **overrides)
