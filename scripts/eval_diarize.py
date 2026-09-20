#!/usr/bin/env python3
"""Manual diarization check on one recording (no STT/MT keys needed).

Usage:
    uv run scripts/eval_diarize.py meeting.m4a
    uv run scripts/eval_diarize.py meeting.m4a --speakers 3 --expect-speakers 3

Diarizes the normalized audio directly, then prints the turn table plus
per-speaker speaking time for eyeballing. With --expect-speakers N, exits
non-zero when fewer than N distinct speakers are found or one speaker owns
everything (crude collapse detector).
"""

import argparse
import tempfile
import time
from pathlib import Path

from omnilingual.config import load_settings
from omnilingual.diarize import build_diarizer
from omnilingual.diarize.base import Turn
from omnilingual.pipeline import prepare


def summarize(turns: list[Turn]) -> dict[str, float]:
    """Speaking seconds per raw engine label."""
    totals: dict[str, float] = {}
    for t in turns:
        totals[t.speaker] = totals.get(t.speaker, 0.0) + max(
            0.0, t.end_s - t.start_s
        )
    return totals


def check_collapse(turns: list[Turn], expected: int) -> str | None:
    """None when healthy, otherwise a human-readable failure reason."""
    totals = summarize(turns)
    if not totals:
        return "no speech segments found"
    if len(totals) < expected:
        return f"found {len(totals)} speaker(s), expected {expected}"
    total = sum(totals.values()) or 1.0
    top, top_s = max(totals.items(), key=lambda kv: kv[1])
    if top_s / total >= 0.999 and expected > 1:
        return f"speaker {top} owns 100% of a multi-speaker recording"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recording", type=Path, help="any ffmpeg-readable recording")
    ap.add_argument("--speakers", type=int, default=None, help="known speaker count hint")
    ap.add_argument("--expect-speakers", type=int, default=None, help="fail unless this many found")
    ap.add_argument("--work-dir", type=Path, default=None, help="reuse a work dir (default: temp)")
    args = ap.parse_args()

    if not args.recording.is_file():
        ap.error(f"no such file: {args.recording}")

    settings = load_settings(diarizer="sherpa", num_speakers=args.speakers)
    try:
        dz = build_diarizer(settings)
    except Exception as exc:
        print(f"unavailable: {exc}")
        return 2

    if args.work_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="eval-diarize-")
        work_dir = Path(tmp.name)
    else:
        tmp = None
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
    try:
        _duration, _chunks = prepare(args.recording, work_dir, settings)
        wav = work_dir / "normalized.wav"
        t0 = time.monotonic()
        turns = dz.diarize(wav)
    finally:
        if tmp is not None:
            tmp.cleanup()
    dt = time.monotonic() - t0

    print(f"model={dz.model}  turns={len(turns)}  {dt:.1f}s")
    for t in turns:
        print(f"  [{t.start_s:7.1f} → {t.end_s:7.1f}]  speaker {t.speaker}")
    print("share:")
    total = sum(summarize(turns).values()) or 1.0
    for speaker, seconds in sorted(
        summarize(turns).items(), key=lambda kv: -kv[1]
    ):
        print(f"  speaker {speaker}: {seconds:.1f}s ({100 * seconds / total:.0f}%)")

    if args.expect_speakers is not None:
        failure = check_collapse(turns, args.expect_speakers)
        if failure is not None:
            print(f"COLLAPSE: {failure}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
