"""Sarvam Saaras speech-to-text over the synchronous REST endpoint (<30 s audio)."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import httpx

from omnilingual.config import Settings
from omnilingual.http import send_with_retry
from omnilingual.models import STTResult


class SarvamSTT:
    mode = "transcribe"

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0))
        self._sleep = sleep
        self.model = settings.stt_model

    def transcribe(self, wav_path: Path) -> STTResult:
        key = self._settings.require_key()
        url = f"{self._settings.base_url}/speech-to-text"
        audio = wav_path.read_bytes()

        def send() -> httpx.Response:
            return self._client.post(
                url,
                headers={"api-subscription-key": key},
                files={"file": (wav_path.name, audio, "audio/wav")},
                data={"model": self.model, "mode": self.mode, "language_code": "unknown"},
            )

        resp = send_with_retry(send, sleep=self._sleep)
        body = resp.json()
        return STTResult(
            lang=body.get("language_code") or "unknown",
            prob=float(body.get("language_probability") or 0.0),
            text=(body.get("transcript") or "").strip(),
        )
