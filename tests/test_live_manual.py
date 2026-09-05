"""Manual live-session checklist (spec §10). NOT run by default.

Run on a Mac with mic + BlackHole + SARVAM_API_KEY set::

    OMNILINGUAL_MANUAL=1 uv run pytest tests/test_live_manual.py -q

Marks:
1. Mic loopback (30 s): `uv run omnilingual live --out loop.md`, speak Hindi
   ~30 s, Ctrl+C once. Expect: 2-4 segments, Hindi text + English quotes,
   exit 0, `tail -f loop.md` grew live.
2. One real STT+MT round trip is covered by mark 1 (no mocks involved).
3. TCC prompt path: `tccutil reset Microphone`, re-run mark 1, expect the
   Settings ▸ Privacy ▸ Microphone hint and exit 1 (no traceback).
4. BlackHole clip diff: play a fixed clip through BlackHole twice — once via
   `live --mic-only` off (Aggregate) into live.md, once transcribed from a
   file recording. Expect: same sentences, timestamps within ±2 s.
5. 48 kHz mismatch: set BlackHole to 48 kHz in Audio MIDI Setup, run
   `--check-audio`. Expect: probe still reports a dBFS level (aresample
   absorbs it) or a clear device error — never silent garbage.
"""

import os

import pytest
from typer.testing import CliRunner

from omnilingual.cli import app

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("OMNILINGUAL_MANUAL") != "1",
        reason="manual only: needs mic, BlackHole, and SARVAM_API_KEY"),
]

runner = CliRunner()


def test_check_audio_smoke():
    res = runner.invoke(app, ["live", "--out", "manual.md", "--check-audio"])
    assert res.exit_code == 0, res.output
    assert "dBFS" in res.output
