"""Google Gemini text translation to English (AI Studio free tier)."""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx

from omnilingual.config import Settings
from omnilingual.http import SarvamError, send_with_retry
from omnilingual.translate.mayura import split_text

# Gemini names languages in English; the pipeline speaks Sarvam BCP-47 tags.
LANG_NAMES: dict[str, str] = {
    "as-IN": "Assamese",
    "bn-IN": "Bengali",
    "brx-IN": "Bodo",
    "doi-IN": "Dogri",
    "en-IN": "English",
    "gu-IN": "Gujarati",
    "hi-IN": "Hindi",
    "kn-IN": "Kannada",
    "kok-IN": "Konkani",
    "ks-IN": "Kashmiri",
    "mai-IN": "Maithili",
    "ml-IN": "Malayalam",
    "mni-IN": "Manipuri",
    "mr-IN": "Marathi",
    "ne-IN": "Nepali",
    "od-IN": "Odia",
    "pa-IN": "Punjabi",
    "sa-IN": "Sanskrit",
    "sat-IN": "Santali",
    "sd-IN": "Sindhi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "ur-IN": "Urdu",
}

SYSTEM_INSTRUCTION = (
    "You are a translation engine for meeting transcripts. Translate the user's "
    "text into English. Output only the translation: no preamble, no notes, no "
    "quotes, no explanations. Keep proper nouns, technical terms, numbers and "
    "code-mixed English words as they appear in the source."
)


class GeminiTranslator:
    # Gemini is free inside the AI Studio quota, so translation adds nothing to
    # the rupee estimate; the pipeline reads this via getattr like inr_per_hour.
    inr_per_10k_chars = 0.0

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0))
        self._sleep = sleep
        self.model = f"gemini:{settings.resolved_mt_model}"

    def supports(self, lang: str) -> bool:
        return lang in LANG_NAMES

    def _generate(self, prompt: str) -> str:
        key = self._settings.require_gemini_key()
        repo_id = self._settings.resolved_mt_model
        url = f"{self._settings.gemini_base_url}/models/{repo_id}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }

        def send() -> httpx.Response:
            return self._client.post(url, headers={"x-goog-api-key": key}, json=payload)

        resp = send_with_retry(send, sleep=self._sleep)
        try:
            body = resp.json()
        except ValueError as exc:
            raise SarvamError("non-JSON response body", resp.status_code, resp.text[:200]) from exc

        candidates = body.get("candidates") or []
        if not candidates:
            block = body.get("promptFeedback", {}).get("blockReason")
            raise SarvamError(
                f"Gemini returned no translation{f' (blocked: {block})' if block else ''}",
                resp.status_code,
                resp.text[:200],
            )
        parts = candidates[0].get("content", {}).get("parts") or []
        return "".join(p["text"] for p in parts if p.get("text")).strip()

    def to_english(self, text: str, src_lang: str) -> str:
        name = LANG_NAMES.get(src_lang, src_lang)
        pieces = split_text(text, self._settings.mt_char_limit)
        return " ".join(self._generate(f"Language: {name}\n\n{p}") for p in pieces if p.strip())