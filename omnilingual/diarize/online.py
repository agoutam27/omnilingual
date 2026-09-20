"""Online speaker tracking for live sessions.

Unlike batch diarization (cluster everything at the end), the live loop must
label each sealed chunk the moment it arrives. This tracker keeps one centroid
vector per speaker heard so far and assigns each new chunk embedding to the
nearest centroid — or mints a new speaker when nothing is close. Numbering is
first-appearance order, so the first voice heard is always "Speaker 1".
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from pathlib import Path

Extractor = Callable[[Path], object]


def _as_floats(vec: object) -> tuple[float, ...]:
    return tuple(float(x) for x in vec)  # type: ignore[union-attr]


def _normalize(vec: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0.0:
        return vec
    return tuple(x / norm for x in vec)


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


class OnlineSpeakerTracker:
    """Incremental centroid clustering over per-chunk speaker embeddings.

    extractor maps a chunk wav to an embedding vector (any 1-D sequence of
    numbers; sherpa_onnx returns numpy arrays, which work as-is).
    """

    def __init__(
        self,
        extractor: Extractor,
        *,
        threshold: float = 0.6,
        floor: float = 0.45,
        margin: float = 0.05,
        update: float = 0.2,
        max_speakers: int = 32,
    ) -> None:
        self._extractor = extractor
        self._threshold = threshold
        self._floor = floor
        self._margin = margin
        self._update = update
        self._max_speakers = max_speakers
        self._centroids: list[tuple[float, ...]] = []
        # Live mode's stt_workers threads share this tracker; centroid
        # reads and updates must be atomic so two threads never mint the
        # same "Speaker N" twice.
        self._lock = threading.Lock()

    @property
    def num_speakers(self) -> int:
        with self._lock:
            return len(self._centroids)

    def assign(self, wav_path: Path) -> str:
        # Extraction runs inside the lock: live stt_workers share one
        # tracker, and serializing end-to-end keeps label minting atomic.
        with self._lock:
            embedding = _normalize(_as_floats(self._extractor(wav_path)))
            best_idx = -1
            best_score = -1.0
            for i, centroid in enumerate(self._centroids):
                score = _cosine(embedding, centroid)
                if score > best_score:
                    best_idx = i
                    best_score = score
            if best_idx >= 0 and (
                best_score >= self._threshold
                or (
                    best_score >= self._floor
                    and best_score >= self._threshold - self._margin
                )
            ):
                centroid = self._centroids[best_idx]
                moved = tuple(
                    (1.0 - self._update) * c + self._update * e
                    for c, e in zip(centroid, embedding)
                )
                self._centroids[best_idx] = _normalize(moved)
                return f"Speaker {best_idx + 1}"
            if len(self._centroids) < self._max_speakers:
                self._centroids.append(embedding)
                return f"Speaker {len(self._centroids)}"
            # Capped: fall back to the nearest centroid rather than growing
            # unboundedly on a long, noisy meeting.
            return f"Speaker {best_idx + 1}" if best_idx >= 0 else "Speaker 1"
