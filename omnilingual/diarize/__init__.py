"""Speaker diarizer factory: the single place that turns Settings into a diarizer."""

from __future__ import annotations

from omnilingual.config import ConfigError, Settings
from omnilingual.diarize.base import Diarizer


def build_diarizer(settings: Settings) -> Diarizer:
    if settings.diarizer == "sherpa":
        from omnilingual.diarize.sherpa import SherpaDiarizer

        return SherpaDiarizer(settings)

    raise ConfigError(
        f"unknown diarizer {settings.diarizer!r} (choose from: sherpa)"
    )
