#!/usr/bin/env python3
"""Manual A/B check of the STT providers on one audio chunk.

Usage:
    uv run scripts/eval_stt.py .omnilingual/<hash>/chunks/0007.wav
    uv run scripts/eval_stt.py chunk.wav --providers sarvam,groq

Prints each provider's model, detected language, confidence, wall time, and
transcript. Credentials come from the environment (SARVAM_API_KEY,
GROQ_API_KEY) as usual; a provider that is unavailable (missing key or extra)
is reported and skipped, not fatal. Feed it a chunk from a work directory so
every provider sees the exact same 16 kHz mono audio the pipeline would send.
"""

import argparse
import time
from pathlib import Path

from omnilingual.config import STT_PROVIDERS, load_settings
from omnilingual.stt import build_stt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav", type=Path, help="WAV chunk to transcribe")
    ap.add_argument("--providers", default=",".join(STT_PROVIDERS),
                    help="comma-separated subset (default: all)")
    args = ap.parse_args()

    if not args.wav.is_file():
        ap.error(f"no such file: {args.wav}")

    for name in [p.strip() for p in args.providers.split(",") if p.strip()]:
        try:
            stt = build_stt(load_settings(stt_provider=name))
        except Exception as exc:
            print(f"{name:<12}  unavailable: {exc}")
            continue
        t0 = time.monotonic()
        try:
            r = stt.transcribe(args.wav)
        except Exception as exc:
            print(f"{name:<12}  error: {exc}")
            continue
        dt = time.monotonic() - t0
        print(f"{name:<12}  model={stt.model}")
        print(f"{'':<12}  lang={r.lang} p={r.prob:.2f} {dt:.1f}s")
        print(f"{'':<12}  {r.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
