"""Translation provider factory: the single place that turns Settings into a Translator."""

from __future__ import annotations

from omnilingual.config import Settings
from omnilingual.translate.base import Translator


def build_translator(settings: Settings) -> Translator:
    if settings.mt_provider == "gemini":
        from omnilingual.translate.gemini import GeminiTranslator

        return GeminiTranslator(settings)

    if settings.mt_provider == "indictrans2":
        from omnilingual.translate.indictrans2 import IndicTrans2Translator

        return IndicTrans2Translator(settings)

    from omnilingual.translate.mayura import MayuraTranslator

    return MayuraTranslator(settings)