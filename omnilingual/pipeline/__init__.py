"""Wire normalize → chunk → STT → translate → Transcript, with caching and resumability."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import wave
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from omnilingual.audio.chunker import chunk_audio
from omnilingual.audio.normalize import normalize, probe_duration
from omnilingual.cache import JsonCache, mt_key, stt_key
from omnilingual.config import Settings
from omnilingual.diarize.assign import assign_speakers
from omnilingual.diarize.base import Diarizer, Turn
from omnilingual.http import AuthError, QuotaError, SarvamError, TransientError
from omnilingual.models import Chunk, Cost, Segment, STTResult, Transcript, chunks_from_json, chunks_to_json
from omnilingual.stt.base import STTProvider
from omnilingual.translate.base import Translator

log = logging.getLogger("omnilingual")

Progress = Callable[[int, int, Segment], None]

T = TypeVar("T")

STT_FAILED_TEXT = "[transcription failed]"

LIVE_SESSION_KIND = "omnilingual-live-session"


def work_dir_for(source: Path, root: Path) -> Path:
    with source.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()[:12]
    return root / digest


def _chunking_signature(settings: Settings) -> dict[str, float]:
    return {"max_chunk_s": settings.max_chunk_s, "min_chunk_s": settings.min_chunk_s}


def _read_json(path: Path) -> object | None:
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


def _price(audio_seconds: float, mt_chars: int, settings: Settings, stt: STTProvider | None = None) -> float:
    rate = getattr(stt, "inr_per_hour", None)
    stt_cost = audio_seconds / 3600.0 * (rate if rate is not None else settings.stt_inr_per_hour)
    mt = mt_chars / 10_000.0 * settings.mt_inr_per_10k_chars
    return round(stt_cost + mt, 4)


def estimate(
    duration_s: float,
    chunks: list[Chunk],
    settings: Settings,
    chars_per_second: float = 15.0,
    stt: STTProvider | None = None,
) -> Cost:
    audio_seconds = sum(c.duration_s for c in chunks) or duration_s
    mt_chars = int(audio_seconds * chars_per_second)
    return Cost(audio_seconds=audio_seconds, mt_chars=mt_chars, inr_estimate=_price(audio_seconds, mt_chars, settings, stt))


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


def _diarize_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def _diarize_cached(
    wav_path: Path, diarizer: Diarizer, cache: JsonCache, num_speakers: int | None
) -> list[Turn]:
    """Diarize once per audio file. The work dir is already content-addressed
    by the recording, so the key only needs the model and speaker count."""
    key = f"diarize-{_diarize_slug(diarizer.model)}-ns{num_speakers or 'auto'}"
    return _cached_call(
        cache,
        "diarize",
        key,
        call=lambda: diarizer.diarize(wav_path),
        decode=lambda hit: [Turn(**d) for d in hit["turns"]],
        encode=lambda turns: {
            "turns": [
                {"start_s": t.start_s, "end_s": t.end_s, "speaker": t.speaker}
                for t in turns
            ]
        },
        describe=f"{wav_path.name} diarization",
    ) or []


def _combine_session_wavs(session_dir: Path, chunks: list[Chunk]) -> Path:
    """Concatenate sealed live chunks (all 16 kHz mono s16le) into one audio
    file so the offline diarizer sees the whole session contiguously."""
    out = session_dir / "diarize-input.wav"
    with wave.open(str(out), "wb") as wout:
        wout.setnchannels(1)
        wout.setsampwidth(2)
        wout.setframerate(16000)
        for chunk in chunks:
            with wave.open(str(chunk.wav_path), "rb") as win:
                if (win.getnchannels(), win.getsampwidth(), win.getframerate()) != (1, 2, 16000):
                    raise ValueError(f"chunk wav has unexpected format: {chunk.wav_path}")
                wout.writeframes(win.readframes(win.getnframes()))
    return out


def transcribe_chunks(
    chunks: list[Chunk],
    duration_s: float,
    source: Path,
    settings: Settings,
    stt: STTProvider,
    translator: Translator,
    cache: JsonCache,
    progress: Progress | None = None,
    speakers: dict[int, str | None] | None = None,
) -> Transcript:
    """The batch STT → translate loop over ready-made chunks. Shared by the file
    pipeline (run), live-session recovery (run_from_chunks), and nothing else."""
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
                # Silence belongs to nobody, so it never carries a speaker label.
                seg = Segment(chunk, result.lang, result.prob, "", None, "no_speech")
            else:
                seg = Segment(chunk, result.lang, result.prob, result.text, None, "ok")
                if speakers is not None:
                    seg.speaker = speakers.get(chunk.idx)
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

    cost = Cost(audio_seconds=duration_s, mt_chars=mt_chars, inr_estimate=_price(duration_s, mt_chars, settings, stt))
    return Transcript(source=source, duration_s=duration_s, segments=segments, cost=cost)


def _speakers_for_chunks(
    chunks: list[Chunk],
    audio: Path,
    settings: Settings,
    diarizer: Diarizer | None,
    cache: JsonCache,
) -> dict[int, str | None] | None:
    """Diarize one audio file and map the turns onto chunks. None when off."""
    if diarizer is None:
        return None
    turns = _diarize_cached(audio, diarizer, cache, settings.num_speakers)
    return assign_speakers(chunks, turns)


def run(
    source: Path,
    work_dir: Path,
    settings: Settings,
    stt: STTProvider,
    translator: Translator,
    cache: JsonCache,
    progress: Progress | None = None,
    diarizer: Diarizer | None = None,
) -> Transcript:
    duration, chunks = prepare(source, work_dir, settings)
    speakers = _speakers_for_chunks(chunks, work_dir / "normalized.wav", settings, diarizer, cache)
    return transcribe_chunks(chunks, duration, source, settings, stt, translator, cache, progress, speakers=speakers)


def run_from_chunks(
    session_dir: Path,
    settings: Settings,
    stt: STTProvider,
    translator: Translator,
    cache: JsonCache,
    progress: Progress | None = None,
    diarizer: Diarizer | None = None,
) -> Transcript:
    """Finish a live session's sealed chunks with the batch loop.

    The session's own chunks.json is the manifest and its cache/ dir is the
    cache namespace, so chunks already transcribed during the live run cost
    nothing here; only the chunks the live run never billed are paid for.
    """
    session = _read_json(session_dir / "session.json")
    if not isinstance(session, dict) or session.get("kind") != LIVE_SESSION_KIND:
        raise ValueError(f"not a live session dir (no {LIVE_SESSION_KIND} session.json): {session_dir}")
    manifest = session_dir / "chunks.json"
    try:
        chunks = chunks_from_json(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"unusable chunk manifest in {session_dir}: {exc}") from exc
    if not chunks:
        raise ValueError(f"live session has no sealed chunks yet: {session_dir}")
    missing = [c.wav_path for c in chunks if not c.wav_path.exists()]
    if missing:
        raise ValueError(f"live session missing {len(missing)} chunk wav(s), e.g. {missing[0]}")
    speakers = None
    if diarizer is not None:
        combined = _combine_session_wavs(session_dir, chunks)
        speakers = _speakers_for_chunks(chunks, combined, settings, diarizer, cache)
    return transcribe_chunks(chunks, chunks[-1].end_s, Path(session_dir.name), settings, stt, translator, cache, progress, speakers=speakers)
