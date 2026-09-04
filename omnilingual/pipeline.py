"""Wire normalize → chunk → STT → translate → Transcript, with caching and resumability."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from omnilingual.audio.chunker import chunk_audio
from omnilingual.audio.normalize import normalize, probe_duration
from omnilingual.cache import JsonCache, mt_key, stt_key
from omnilingual.config import Settings
from omnilingual.http import AuthError, QuotaError, SarvamError, TransientError
from omnilingual.models import Chunk, Cost, Segment, STTResult, Transcript, chunks_from_json, chunks_to_json
from omnilingual.stt.base import STTProvider
from omnilingual.translate.base import Translator

log = logging.getLogger("omnilingual")

Progress = Callable[[int, int, Segment], None]

T = TypeVar("T")

STT_FAILED_TEXT = "[transcription failed]"


def work_dir_for(source: Path, root: Path) -> Path:
    with source.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()[:12]
    return root / digest


def _chunking_signature(settings: Settings) -> dict[str, float]:
    return {"max_chunk_s": settings.max_chunk_s, "min_chunk_s": settings.min_chunk_s}


def _read_json(path: Path):
    """Parsed JSON, or None when the file is absent or unusable. A corrupt work-dir
    file means "redo the work", never a crash."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _reusable_chunks(manifest: Path, chunking: Path, settings: Settings) -> list[Chunk] | None:
    """Chunks from a previous run, if they are still valid: the manifest parses, every
    chunk file is present, and the chunk settings that produced them are unchanged."""
    if _read_json(chunking) != _chunking_signature(settings):
        return None
    try:
        chunks = chunks_from_json(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
        return None
    return chunks if all(c.wav_path.exists() for c in chunks) else None


def prepare(source: Path, work_dir: Path, settings: Settings) -> tuple[float, list[Chunk]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    wav = work_dir / "normalized.wav"
    manifest = work_dir / "chunks.json"
    chunking = work_dir / "chunking.json"
    chunks_dir = work_dir / "chunks"

    duration = probe_duration(wav) if wav.exists() else normalize(source, wav)

    chunks = _reusable_chunks(manifest, chunking, settings)
    if chunks is not None:
        return duration, chunks

    # Fewer chunks than last time would leave orphaned wavs behind, and content-addressed
    # cache keys make a stale file harmless but never reclaimed. Start the directory clean.
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks = chunk_audio(wav, chunks_dir, settings.max_chunk_s, settings.min_chunk_s)
    manifest.write_text(chunks_to_json(chunks), encoding="utf-8")
    chunking.write_text(json.dumps(_chunking_signature(settings), indent=2), encoding="utf-8")
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


def _cached_call(
    cache: JsonCache,
    namespace: str,
    key: str,
    call: Callable[[], T],
    decode: Callable[[dict], T],
    encode: Callable[[T], dict],
    *,
    describe: str,
) -> T | None:
    """Run `call` unless the cache already answers `key`. None means the API call
    failed non-fatally and this segment has no result for this run.

    Only permanent failures are remembered. A TransientError has already exhausted
    its retries, but the cause (rate limit, outage, flaky network) is expected to
    clear, so nothing is written and the next run is free to try again. Keeping this
    policy in one place is the point: STT and MT must not drift apart on it.
    """
    hit = cache.get(namespace, key)
    if hit is not None:
        return None if hit.get("failed") else decode(hit)
    try:
        value = call()
    except (AuthError, QuotaError):
        raise
    except TransientError as exc:
        log.error("%s failed transiently, will retry on the next run: %s", describe, exc)
        return None
    except SarvamError as exc:
        log.error("%s failed: %s", describe, exc)
        cache.put(namespace, key, {"failed": True, "error": str(exc)})
        return None
    cache.put(namespace, key, encode(value))
    return value


def _stt_cached(chunk: Chunk, stt: STTProvider, cache: JsonCache) -> STTResult | None:
    """Return STT result, using cache. None means STT failed non-fatally."""
    key = stt_key(chunk.wav_path.read_bytes(), stt.model, stt.mode)
    return _cached_call(
        cache,
        "stt",
        key,
        call=lambda: stt.transcribe(chunk.wav_path),
        decode=lambda hit: STTResult(lang=hit["lang"], prob=hit["prob"], text=hit["text"]),
        encode=lambda r: {"lang": r.lang, "prob": r.prob, "text": r.text},
        describe=f"chunk {chunk.idx} STT",
    )


def _mt_cached(text: str, lang: str, translator: Translator, cache: JsonCache) -> str | None:
    key = mt_key(text, lang, "en-IN", translator.model)
    return _cached_call(
        cache,
        "mt",
        key,
        call=lambda: translator.to_english(text, lang),
        decode=lambda hit: hit["english"],
        encode=lambda english: {"english": english},
        describe=f"translation of {lang}",
    )


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
            if not result.text.strip():
                # A successful call that found no speech: silence, music, or crosstalk.
                # Nothing to translate and nothing to show but the note.
                seg = Segment(chunk, result.lang, result.prob, "", None, "no_speech")
            else:
                seg = Segment(chunk, result.lang, result.prob, result.text, None, "ok")
                if result.lang != "en-IN":
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
