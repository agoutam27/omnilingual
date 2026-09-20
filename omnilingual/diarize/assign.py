"""Map diarization turns onto pipeline chunks.

A chunk can straddle a speaker change, so each chunk takes the turn with the
largest temporal overlap; exact ties keep the previous chunk's speaker.
Raw engine labels ("0", "7", ...) are renumbered to first-appearance
"Speaker N" tags so output is stable and human-readable.
"""

from __future__ import annotations

from omnilingual.diarize.base import Turn
from omnilingual.models import Chunk


def _overlap(chunk: Chunk, turn: Turn) -> float:
    return max(0.0, min(chunk.end_s, turn.end_s) - max(chunk.start_s, turn.start_s))


def renumber(raw: list[str | None]) -> list[str | None]:
    """First-appearance order: ["4", "2", "4"] -> ["Speaker 1", "Speaker 2", "Speaker 1"]."""
    order: dict[str, str] = {}
    out: list[str | None] = []
    for label in raw:
        if label is None:
            out.append(None)
            continue
        if label not in order:
            order[label] = f"Speaker {len(order) + 1}"
        out.append(order[label])
    return out


_EPS = 1e-9


def assign_speakers(
    chunks: list[Chunk], turns: list[Turn]
) -> dict[int, str | None]:
    """Dominant speaker per chunk, keyed by chunk idx."""
    raw: list[str | None] = []
    for chunk in chunks:
        scored = sorted(
            ((_overlap(chunk, turn), turn.speaker) for turn in turns),
            key=lambda kv: -kv[0],
        )
        scored = [(overlap, speaker) for overlap, speaker in scored if overlap > 0.0]
        if not scored:
            raw.append(None)
            continue
        top = scored[0][0]
        tied = {speaker for overlap, speaker in scored if abs(overlap - top) <= _EPS}
        if len(tied) > 1 and raw and raw[-1] in tied:
            raw.append(raw[-1])
        else:
            raw.append(scored[0][1])
    return {chunk.idx: label for chunk, label in zip(chunks, renumber(raw))}
