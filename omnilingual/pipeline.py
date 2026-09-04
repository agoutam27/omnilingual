"""Wire normalize → chunk → STT → translate → Transcript, with caching and resumability."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

from omnilingual.audio.chunker import chunk_audio
from omnilingual.audio.normalize import normalize, probe_duration
from omnilingual.cache import JsonCache, mt_key, stt_key
from omnilingual.config import Settings
from omnilingual.http import AuthError, QuotaError, SarvamError
from omnilingual.models import Chunk, Cost, Segment, STTResult, Transcript, chunks_from_json, chunks_to_json
from omnilingual.stt.base import STTProvider
from omnilingual.translate.base import Translator

log = logging.getLogger("omnilingual")

Progress = Callable[[int, int, Segment], None]

STT_FAILED_TEXT = "[transcription failed]"


def work_dir_for(source: Path, root: Path) -> Path:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    return root / digest


def prepare(source: Path, work_dir: Path, settings: Settings) -> tuple[float, list[Chunk]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    wav = work_dir / "normalized.wav"
    manifest = work_dir / "chunks.json"

    if wav.exists():
        duration = probe_duration(wav)
    else:
        duration = normalize(source, wav)

    if manifest.exists():
        chunks = chunks_from_json(manifest.read_text(encoding="utf-8"))
        if all(c.wav_path.exists() for c in chunks):
            return duration, chunks

    chunks = chunk_audio(wav, work_dir / "chunks", settings.max_chunk_s, settings.min_chunk_s)
    manifest.write_text(chunks_to_json(chunks), encoding="utf-8")
    return duration, chunks


def _price(audio_seconds: float, mt_chars: int, settings: Settings) -> float:
    stt = audio_seconds / 3600.0 * settings.stt_inr_per_hour
    mt = mt_chars / 10_000.0 * settings.mt_inr_per_10k_chars
    return round(stt + mt, 4)


def estimate(
    duration_s: float, chunks: list[Chunk], settings: Settings, chars_per_second: float = 15.0
) -> Cost:
    audio_seconds = sum(c.duration_s for c in chunks) or duration_s
    mt_chars = int(audio_seconds * chars_per_second)
    return Cost(audio_seconds=audio_seconds, mt_chars=mt_chars, inr_estimate=_price(audio_seconds, mt_chars, settings))


def _stt_cached(chunk: Chunk, stt: STTProvider, cache: JsonCache) -> STTResult | None:
    """Return STT result, using cache. None means STT failed non-fatally."""
    wav_bytes = chunk.wav_path.read_bytes()
    key = stt_key(wav_bytes, stt.model, stt.mode)
    hit = cache.get("stt", key)
    if hit is not None:
        if hit.get("failed"):
            return None
        return STTResult(lang=hit["lang"], prob=hit["prob"], text=hit["text"])
    try:
        result = stt.transcribe(chunk.wav_path)
    except (AuthError, QuotaError):
        raise
    except SarvamError as exc:
        log.error("chunk %d STT failed: %s", chunk.idx, exc)
        cache.put("stt", key, {"failed": True, "error": str(exc)})
        return None
    cache.put("stt", key, {"lang": result.lang, "prob": result.prob, "text": result.text})
    return result


def _mt_cached(text: str, lang: str, translator: Translator, cache: JsonCache) -> str | None:
    key = mt_key(text, lang, "en-IN", translator.model)
    hit = cache.get("mt", key)
    if hit is not None:
        return None if hit.get("failed") else hit["english"]
    try:
        english = translator.to_english(text, lang)
    except (AuthError, QuotaError):
        raise
    except SarvamError as exc:
        log.error("translation failed for %s: %s", lang, exc)
        cache.put("mt", key, {"failed": True, "error": str(exc)})
        return None
    cache.put("mt", key, {"english": english})
    return english


def run(
    source: Path,
    work_dir: Path,
    settings: Settings,
    stt: STTProvider,
    translator: Translator,
    cache: JsonCache,
    progress: Progress | None = None,
) -> Transcript:
    duration, chunks = prepare(source, work_dir, settings)
    segments: list[Segment] = []
    mt_chars = 0

    for i, chunk in enumerate(chunks, start=1):
        result = _stt_cached(chunk, stt, cache)
        if result is None:
            seg = Segment(chunk, "unknown", 0.0, STT_FAILED_TEXT, None, "stt_failed")
        else:
            if settings.langs and result.lang not in settings.langs:
                log.warning(
                    "chunk %d: detected %s (p=%.2f) outside configured languages %s",
                    chunk.idx, result.lang, result.prob, ",".join(settings.langs),
                )
            seg = Segment(chunk, result.lang, result.prob, result.text, None, "ok")
            if result.text and result.lang != "en-IN":
                if not translator.supports(result.lang):
                    seg.status = "mt_unsupported"
                else:
                    english = _mt_cached(result.text, result.lang, translator, cache)
                    if english is None:
                        seg.status = "mt_failed"
                    else:
                        seg.english = english
                        mt_chars += len(result.text)
        segments.append(seg)
        if progress:
            progress(i, len(chunks), seg)

    cost = Cost(audio_seconds=duration, mt_chars=mt_chars, inr_estimate=_price(duration, mt_chars, settings))
    return Transcript(source=source, duration_s=duration, segments=segments, cost=cost)
