"""Live session pipeline: per-chunk STT/MT classification plus calibration."""

from __future__ import annotations

import logging

from omnilingual.cache import JsonCache
from omnilingual.config import Settings
from omnilingual.models import Chunk, Segment
from omnilingual.pipeline import (
    STT_FAILED_TEXT,
    _mt_cached,
    _price,
    _stt_cached,
)

log = logging.getLogger("omnilingual")


def calibrate_energy_floor(ambient_rms: float) -> float:
    """Speech floor from the 1 s ambient probe: 4x ambient, floor 0.004."""
    return max(ambient_rms * 4.0, 0.004)


def process_chunk(
    chunk: Chunk,
    *,
    speech: bool,
    stt,
    translator,
    cache: JsonCache,
    settings: Settings,
) -> tuple[Segment, float, bool]:
    """Transcribe + translate one sealed chunk. Returns (segment, inr, billed).

    Mirrors the batch per-chunk classification exactly. QuotaError/AuthError
    propagate (the run loop converts them into a capture-while-halted state).
    """
    if not speech:
        seg = Segment(chunk=chunk, lang="", prob=0.0, text="",
                      english=None, status="no_speech")
        return (seg, 0.0, False)
    result = _stt_cached(chunk, stt, cache)
    if result is None:
        seg = Segment(chunk=chunk, lang="unknown", prob=0.0,
                      text=STT_FAILED_TEXT, english=None, status="stt_failed")
        return (seg, _price(chunk.duration_s, 0, settings), True)
    if settings.langs and result.lang not in settings.langs:
        log.warning(
            "chunk %d: detected %s (p=%.2f) outside configured languages %s",
            chunk.idx, result.lang, result.prob, ",".join(settings.langs),
        )
    if not result.text.strip():
        seg = Segment(chunk=chunk, lang=result.lang, prob=result.prob,
                      text="", english=None, status="no_speech")
        return (seg, _price(chunk.duration_s, 0, settings), True)
    if result.lang == "en-IN":
        seg = Segment(chunk=chunk, lang="en-IN", prob=result.prob,
                      text=result.text, english=None, status="ok")
        return (seg, _price(chunk.duration_s, 0, settings), True)
    if not translator.supports(result.lang):
        seg = Segment(chunk=chunk, lang=result.lang, prob=result.prob,
                      text=result.text, english=None, status="mt_unsupported")
        return (seg, _price(chunk.duration_s, 0, settings), True)
    english = _mt_cached(result.text, result.lang, translator, cache)
    if english is None:
        seg = Segment(chunk=chunk, lang=result.lang, prob=result.prob,
                      text=result.text, english=None, status="mt_failed")
        return (seg, _price(chunk.duration_s, 0, settings), True)
    seg = Segment(chunk=chunk, lang=result.lang, prob=result.prob,
                  text=result.text, english=english, status="ok")
    return (seg, _price(chunk.duration_s, len(result.text), settings), True)
