"""Groq-hosted Whisper speech-to-text: the cheap cloud fallback (~$0.04/hr for turbo)."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import httpx

from omnilingual.config import Settings
from omnilingual.http import SarvamError, send_with_retry
from omnilingual.models import STTResult
from omnilingual.stt.langs import to_bcp47

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


class GroqSTT:
    """OpenAI-compatible multipart transcription. verbose_json carries the
    detected language as a full lowercase name but no probability, so prob is
    always 0.0 (a display-only field downstream)."""

    mode = "transcribe"
    inr_per_hour = 3.4

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0))
        self._sleep = sleep
        self._repo_id = settings.resolved_stt_model
        self.model = f"groq:{self._repo_id}"

    def transcribe(self, wav_path: Path) -> STTResult:
        key = self._settings.require_groq_key()
        audio = wav_path.read_bytes()

        def send() -> httpx.Response:
            return self._client.post(
                GROQ_TRANSCRIPTION_URL,
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (wav_path.name, audio, "audio/wav")},
                data={"model": self._repo_id, "response_format": "verbose_json"},
            )

        resp = send_with_retry(send, sleep=self._sleep)
        try:
            body = resp.json()
        except ValueError as exc:
            raise SarvamError("non-JSON response body", resp.status_code, resp.text[:200]) from exc
        return STTResult(
            lang=to_bcp47(body.get("language") or ""),
            prob=0.0,
            text=(body.get("text") or "").strip(),
        )
