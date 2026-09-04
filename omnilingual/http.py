"""Retry policy and error types shared by all Sarvam API adapters."""

from __future__ import annotations

import random
import time
from collections.abc import Callable

import httpx


class SarvamError(Exception):
    def __init__(self, message: str, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class AuthError(SarvamError):
    """401/403: key missing, invalid, or not permitted."""


class QuotaError(SarvamError):
    """402: credits exhausted. Safe to resume later."""


class TransientError(SarvamError):
    """429/5xx/network failure that persisted through all retries."""


def send_with_retry(
    send: Callable[[], httpx.Response],
    *,
    retries: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> httpx.Response:
    attempt = 0
    while True:
        try:
            resp = send()
        except httpx.TransportError as exc:
            if attempt >= retries:
                raise TransientError(f"network error after {attempt + 1} attempts: {exc}") from exc
            sleep(base_delay * (2**attempt) + jitter())
            attempt += 1
            continue

        status = resp.status_code
        if 200 <= status < 300:
            return resp
        if status in (401, 403):
            raise AuthError(f"authentication failed ({status})", status, resp.text)
        if status == 402:
            raise QuotaError("Sarvam credits exhausted (402)", status, resp.text)
        if status == 429 or status >= 500:
            if attempt >= retries:
                raise TransientError(
                    f"server returned {status} after {attempt + 1} attempts", status, resp.text
                )
            sleep(base_delay * (2**attempt) + jitter())
            attempt += 1
            continue
        raise SarvamError(f"request failed ({status}): {resp.text[:200]}", status, resp.text)
