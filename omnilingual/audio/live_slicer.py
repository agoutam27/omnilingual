"""Rolling byte-clock slicer for live capture.

Pure with respect to the byte clock: every timestamp is bytes / 32000,
never wall clock, so seals are deterministic under pipe buffering.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

from omnilingual.models import Chunk, chunks_to_json

from .live_capture import BYTES_PER_SECOND, rms


@dataclass(frozen=True)
class SealedChunk:
    chunk: Chunk
    speech: bool


class LiveSlicer:
    """Accumulates PCM, seals speech chunks at silence gaps or the max cut."""

    def __init__(self, session_dir: Path, *, target_s: float = 8.0,
                 max_s: float = 28.0, min_s: float = 5.0,
                 energy_floor: float = 0.004) -> None:
        if not min_s <= target_s <= max_s:
            raise ValueError("require min_s <= target_s <= max_s")
        self.session_dir = session_dir
        self.target = target_s
        self.max_s = max_s
        self.min_s = min_s
        self.energy_floor = energy_floor
        self.manifest_path = session_dir / "chunks.json"
        self._wav_dir = session_dir / "live-chunks"
        self._wav_dir.mkdir(parents=True, exist_ok=True)
        self._buf = bytearray()
        self._clock = 0.0  # session seconds consumed from the stream
        self._chunk_start = 0.0
        self._skip_until: float | None = None
        self._gaps: list[tuple[float, float]] = []
        self._next_idx = 0
        self._manifest: list[Chunk] = []
        self._write_manifest()  # chunks.json always exists from birth (possibly [])

    def _buffered_s(self) -> float:
        return len(self._buf) / BYTES_PER_SECOND

    def feed(self, data: bytes) -> list[SealedChunk]:
        """Append stream bytes, dropping any span inside a known silence gap."""
        t0 = self._clock
        t1 = t0 + len(data) / BYTES_PER_SECOND
        self._clock = t1
        if self._skip_until is not None:
            if t1 <= self._skip_until:
                return []
            cut = int((self._skip_until - t0) * BYTES_PER_SECOND)
            data = data[cut:]
            self._skip_until = None
        self._buf += data
        return self._maybe_seal()

    def note_gap(self, start_s: float, end_s: float) -> list[SealedChunk]:
        """Record a completed silence gap; seal immediately if it closes a chunk."""
        if end_s <= start_s:
            return []
        self._gaps.append((start_s, end_s))
        return self._maybe_seal()

    def _maybe_seal(self) -> list[SealedChunk]:
        out: list[SealedChunk] = []
        while True:
            buffered = self._buffered_s()
            if buffered < self.target and buffered < self.max_s:
                return out
            if buffered >= self.max_s:
                sealed = self._seal(self._chunk_start + self.max_s)
                if sealed is not None:
                    out.append(sealed)
                continue
            hit = self._first_gap_at_or_after(self._chunk_start + self.target)
            if hit is None:
                return out
            start, end = hit
            if start > self._chunk_start + buffered:
                return out
            sealed = self._seal(start)
            # Discard gap audio: drop through the gap end, keep the tail.
            drop_through = int((end - self._chunk_start) * BYTES_PER_SECOND)
            del self._buf[: max(0, drop_through)]
            self._chunk_start = end
            self._gaps = [(a, b) for a, b in self._gaps if b > end]
            if sealed is not None:
                out.append(sealed)

    def _first_gap_at_or_after(self, t: float) -> tuple[float, float] | None:
        cands = [(a, b) for a, b in self._gaps if a >= t - 1e-9]
        if not cands:
            return None
        return min(cands, key=lambda g: g[0])

    def _seal(self, end_s: float) -> SealedChunk | None:
        head_len = int((end_s - self._chunk_start) * BYTES_PER_SECOND)
        head = bytes(self._buf[:head_len])
        del self._buf[:head_len]
        start, self._chunk_start = self._chunk_start, end_s
        if (end_s - start) < self.min_s - 1e-9:
            return None
        speech = rms(head) >= self.energy_floor
        wav_path = self._wav_dir / f"{self._next_idx:04d}.wav"
        self._next_idx += 1
        if speech:
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(head)
            chunk = Chunk(idx=len(self._manifest), start_s=start,
                          end_s=end_s, wav_path=wav_path.resolve())
            self._manifest.append(chunk)
            self._write_manifest()
            return SealedChunk(chunk=chunk, speech=True)
        return SealedChunk(
            chunk=Chunk(idx=-1, start_s=start, end_s=end_s, wav_path=wav_path),
            speech=False,
        )

    def _write_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.part")
        tmp.write_text(chunks_to_json(self._manifest), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def flush(self) -> list[SealedChunk]:
        """Seal the trailing partial chunk if it reaches min_s (Ctrl+C path)."""
        if self._buffered_s() < self.min_s - 1e-9:
            self._buf.clear()
            return []
        sealed = self._seal(self._chunk_start + self._buffered_s())
        return [] if sealed is None else [sealed]
