"""Protocol every translation backend satisfies."""

from __future__ import annotations

from typing import Protocol


class Translator(Protocol):
    model: str

    def supports(self, lang: str) -> bool: ...

    def to_english(self, text: str, src_lang: str) -> str: ...
