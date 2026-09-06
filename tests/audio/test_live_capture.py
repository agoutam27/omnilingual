import io
import math
import struct
import subprocess

import pytest

from tests.conftest import raw_pcm

from omnilingual.audio.live_capture import (
    AudioDevice,
    CaptureError,
    LiveCapture,
    build_command,
    dbfs,
    describe_startup_failure,
    downmix_filter,
    parse_devices,
    parse_silence_line,
    read_exact,
    resolve_device,
    rms,
    voice_band_share,
    VOICE_MIN_SHARE,
)

LISTING = """\
[AVFoundation indev @ 0x7f8] AVFoundation video devices:
[AVFoundation indev @ 0x7f8] [0] FaceTime HD Camera
[AVFoundation indev @ 0x7f8] AVFoundation audio devices:
[AVFoundation indev @ 0x7f8] [0] MacBook Pro Microphone
[AVFoundation indev @ 0x7f8] [1] BlackHole 2ch
[AVFoundation indev @ 0x7f8] [2] Omnilingual
"""


def test_parse_devices_reads_audio_section_only():
    devices = parse_devices(LISTING)
    assert devices == [
        AudioDevice(index=0, name="MacBook Pro Microphone"),
        AudioDevice(index=1, name="BlackHole 2ch"),
        AudioDevice(index=2, name="Omnilingual"),
    ]


def test_resolve_device_substring_case_insensitive():
    devices = parse_devices(LISTING)
    assert resolve_device(devices, "omnilingual").index == 2


def test_resolve_device_no_match_lists_available():
    devices = parse_devices(LISTING)
    with pytest.raises(CaptureError, match="Omnilingual"):
        resolve_device(devices, "nope")


def test_resolve_device_ambiguous_lists_candidates():
    devices = parse_devices(LISTING + "[AVFoundation indev @ 0x7f8] [3] Omnilingual Backup\n")
    with pytest.raises(CaptureError, match="Omnilingual Backup"):
        resolve_device(devices, "omnilingual")


def test_downmix_filter_shapes():
    assert downmix_filter(True) == "aresample=16000"
    assert (
        downmix_filter(False)
        == "pan=mono|c0=0.5*c0+0.25*c1+0.25*c2,aresample=16000"
    )


def test_build_command_uses_single_input():
    cmd = build_command(2, False, -35.0)
    assert cmd.count("-i") == 1
    assert ":2" in cmd
    joined = " ".join(cmd)
    assert "asplit=2[pcm][det]" in joined
    assert "silencedetect=noise=-35.0dB:d=0.4" in joined
    assert cmd[-3:-1] == ["-f", "s16le"]
    assert cmd[-1] == "-"  # PCM goes to stdout, consumed by the capture thread


def test_parse_silence_lines():
    assert parse_silence_line("[silencedetect @ x] silence_start: 12.5") == ("start", 12.5)
    assert parse_silence_line("[silencedetect @ x] silence_end: 13.1") == ("end", 13.1)
    assert parse_silence_line("frame=  100 fps=0.0") is None


def test_rms_and_dbfs():
    assert rms(b"\x00" * 3200) == 0.0
    assert dbfs(0.0) == float("-inf")
    tone = struct.pack("<4h", 1000, -1000, 1000, -1000)
    assert rms(tone) == pytest.approx(1000 / 32768)
    assert dbfs(1.0) == pytest.approx(0.0)


def test_read_exact_raises_on_truncated_stream():
    with pytest.raises(CaptureError, match="ended"):
        read_exact(io.BytesIO(b"\x00" * 10), 12)


def test_startup_failure_maps_permission_error_to_settings_hint():
    msg = describe_startup_failure("Omnilingual", "Error: operation not permitted")
    assert "Privacy" in msg and "Microphone" in msg


def test_open_raises_when_device_missing(monkeypatch):
    class Done:
        returncode = 1
        stderr = LISTING
        stdout = ""

    monkeypatch.setattr("omnilingual.audio.live_capture.ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done())
    cap = LiveCapture("nope")
    with pytest.raises(CaptureError, match="Omnilingual"):
        cap.open()


def test_close_without_open_is_safe():
    LiveCapture("Omnilingual").close()


def _sine(freq: float, seconds: float = 1.0, amp: int = 8000, rate: int = 16000) -> bytes:
    n = int(seconds * rate)
    return struct.pack(
        f"<{n}h",
        *(int(amp * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)),
    )


def test_voice_band_share_marks_voice_tone():
    assert voice_band_share(raw_pcm([("tone", 1.0)])) >= VOICE_MIN_SHARE


def test_voice_band_share_rejects_mains_hum():
    assert voice_band_share(_sine(100.0)) < VOICE_MIN_SHARE


def test_voice_band_share_silence_is_zero():
    assert voice_band_share(bytes(32000)) == 0.0
