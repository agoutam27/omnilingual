"""Sarvam text translation (mayura:v1 or sarvam-translate:v1) to English."""

from __future__ import annotations

import re
import time
from collections.abc import Callable

import httpx

from omnilingual.config import Settings
from omnilingual.http import send_with_retry

MAYURA_LANGS: frozenset[str] = frozenset(
    {"bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN"}
)
SARVAM_TRANSLATE_LANGS: frozenset[str] = MAYURA_LANGS | frozenset(
    {"as-IN", "brx-IN", "doi-IN", "kok-IN", "ks-IN", "mai-IN", "mni-IN", "ne-IN", "sa-IN", "sat-IN", "sd-IN", "ur-IN"}
)

# Split after a sentence terminator (Devanagari danda, Latin . ? !, or newline) followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[।.?!\n])\s+")


def split_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        if not sentence:
            continue
        while len(sentence) > limit:  # single sentence longer than limit: hard split
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:limit])
            sentence = sentence[limit:]
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= limit:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


class MayuraTranslator:
    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0))
        self._sleep = sleep
        self.model = settings.mt_model

    def supports(self, lang: str) -> bool:
        langs = MAYURA_LANGS if self.model == "mayura:v1" else SARVAM_TRANSLATE_LANGS
        return lang in langs

    def _translate_piece(self, piece: str, src_lang: str) -> str:
        key = self._settings.require_key()
        url = f"{self._settings.base_url}/translate"
        payload = {
            "input": piece,
            "source_language_code": src_lang,
            "target_language_code": "en-IN",
            "model": self.model,
            "mode": self._settings.mt_mode,
        }

        def send() -> httpx.Response:
            return self._client.post(url, headers={"api-subscription-key": key}, json=payload)

        resp = send_with_retry(send, sleep=self._sleep)
        return (resp.json().get("translated_text") or "").strip()

    def to_english(self, text: str, src_lang: str) -> str:
        pieces = split_text(text, self._settings.mt_char_limit)
        return " ".join(self._translate_piece(p, src_lang) for p in pieces if p.strip())
