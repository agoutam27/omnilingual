"""Is audio capture ready, and the two ways to fix it when it is not.

Wraps scripts/audio-devices.swift — the same helper setup-mac.sh compiles and
drives — rather than reimplementing Aggregate Device creation. Everything here
is read-only except setup() and restart_daemon(), which the UI reaches only from
an explicit button. In particular nothing here ever switches the system output
device: the Multi-Output Device route silently breaks the volume keys, which is
why the current output is reported for diagnosis and never changed.

Stdlib plus omnilingual's own ffmpeg gate, deliberately: this package must stay
importable without the `ui` extra, because the CLI has no web dependency and
tests/test_ui_packaging.py fails any module here that reaches for one.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from omnilingual.audio.normalize import FfmpegMissingError, ensure_ffmpeg

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_SRC = REPO_ROOT / "scripts" / "audio-devices.swift"
DEVICE_NAME = "Omnilingual"
_TIMEOUT = 30

# How AVFoundation words a refusal. Matched case-insensitively against ffmpeg's
# stderr, because it has been seen both as "Permission denied" and as
# "Device not permitted".
_PERMISSION_MARKERS = ("permission denied", "not permitted", "avfoundation:")

_helper: Path | None = None


@dataclass(frozen=True)
class Readiness:
    device: bool
    blackhole: bool
    ffmpeg: bool
    ffprobe: bool
    mic_authorized: bool
    output: str | None
    detail: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all((self.device, self.blackhole, self.ffmpeg, self.ffprobe,
                    self.mic_authorized))

    def as_dict(self) -> dict:
        # "device" is the name here, while Readiness.device says whether it
        # exists: the page shows one and colours the other, and renaming either
        # would break the shape /api/audio already documents.
        return {
            "ok": self.ok,
            "device": DEVICE_NAME,
            "blackhole": self.blackhole,
            "ffmpeg": self.ffmpeg,
            "ffprobe": self.ffprobe,
            "mic_authorized": self.mic_authorized,
            "output": self.output,
            "detail": list(self.detail),
        }


def helper_path() -> Path:
    """Compile audio-devices.swift into a private temp binary, once per process.

    The temp directory is deliberately left behind: the binary is the process's
    working copy of the helper for as long as the UI runs.
    """
    global _helper
    if _helper is not None:
        return _helper
    if not HELPER_SRC.is_file():
        raise FileNotFoundError(f"missing {HELPER_SRC}")
    out = Path(tempfile.mkdtemp(prefix="omni-audio-")) / "audio-devices"
    subprocess.run(["swiftc", "-o", str(out), str(HELPER_SRC)], check=True,
                   capture_output=True, timeout=_TIMEOUT)
    _helper = out
    return out


def device_list() -> str:
    """Stdout of the helper's `list` subcommand.

    One `"<uid> | <name>"` line per audio device, so callers match a device by
    substring against the whole listing rather than parsing a column.
    """
    result = subprocess.run([str(helper_path()), "list"], capture_output=True,
                            text=True, timeout=_TIMEOUT)
    return result.stdout or ""


def _current_output() -> str | None:
    """The device the system plays through right now — reported, never changed.

    Best-effort by design: this is a diagnostic for "why is meeting audio not
    reaching BlackHole", so an absent or stuck SwitchAudioSource must not fail
    the readiness verdict the page is waiting on.
    """
    if shutil.which("SwitchAudioSource") is None:
        return None
    try:
        result = subprocess.run(["SwitchAudioSource", "-c"], capture_output=True,
                                text=True, timeout=_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout or "").strip() or None


def _short_error(text: str) -> str:
    """The last thing ffmpeg said, trimmed to something a page can show."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:200] if lines else "ffmpeg failed without a message"


def _capture_verdict(device: str) -> tuple[bool, str]:
    """Attempt one second of capture and report what actually happened.

    macOS exposes no supported way to read Microphone TCC status, so this is the
    only way to learn it. It works because capture runs through ffmpeg in this
    process rather than the browser's getUserMedia: there is no browser prompt,
    but macOS still refuses at the AVFoundation layer.

    The leading colon in the device string selects the audio section, which is
    the form live_capture.py opens capture with: a bare `-i Omnilingual` asks
    avfoundation for a *video* device of that name and fails "Video device not
    found", which says nothing about the microphone.

    Returns (authorized, reason). `reason` explains a failure and is empty when
    authorized, so the caller can name a permission denial only when one was
    actually observed.
    """
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-f", "avfoundation",
         "-i", f":{device}", "-t", "1", "-f", "null", "-"],
        capture_output=True, text=True, timeout=_TIMEOUT)
    stderr = result.stderr or ""
    if any(marker in stderr.lower() for marker in _PERMISSION_MARKERS):
        return False, ("macOS denied microphone access — approve this app in "
                       "System Settings → Privacy & Security → Microphone")
    if result.returncode != 0:
        # Not a denial: a busy device or a broken aggregate fails the same way.
        return False, _short_error(stderr + (result.stdout or ""))
    return True, ""


def _mic_authorized(device: str) -> bool:
    """Empirically test whether this process may record.

    Real audio I/O for one second, which is why probe() takes mic=False: a page
    poll that cannot ask must not make noise.
    """
    return _capture_verdict(device)[0]


def probe(device: str = DEVICE_NAME, *, mic: bool = True) -> Readiness:
    """Report capture readiness. Read-only, so safe to call at any time."""
    detail: list[str] = []
    ffmpeg = shutil.which("ffmpeg") is not None
    ffprobe = shutil.which("ffprobe") is not None
    if not (ffmpeg and ffprobe):
        # ensure_ffmpeg owns the wording of this failure for the CLI, so borrow
        # its message instead of keeping a second one to drift. probe reports
        # where the CLI raises, which is why it cannot simply call the gate.
        try:
            ensure_ffmpeg()
        except FfmpegMissingError as exc:
            detail.append(str(exc))

    listing = ""
    if ffmpeg and ffprobe:
        try:
            listing = device_list()
        except (OSError, subprocess.SubprocessError) as exc:
            detail.append(f"could not run the audio-device helper: {exc}")

    device_ok = device in listing
    blackhole = "BlackHole" in listing
    if listing and not device_ok:
        detail.append(f"the Aggregate Device '{device}' does not exist yet")
    if listing and not blackhole:
        detail.append("BlackHole 2ch is not installed; "
                      "run: brew install blackhole-2ch")

    mic_ok = False
    if not mic:
        mic_ok = True  # not probed, so do not report a false alarm
    elif ffmpeg and device_ok:
        mic_ok, reason = _capture_verdict(device)
        if not mic_ok:
            detail.append(f"microphone capture failed: {reason}")

    return Readiness(device=device_ok, blackhole=blackhole, ffmpeg=ffmpeg,
                     ffprobe=ffprobe, mic_authorized=mic_ok,
                     output=_current_output(), detail=detail)


def setup(device: str = DEVICE_NAME) -> Iterator[str]:
    """Create the Aggregate Device, yielding progress lines. Idempotent."""
    yield "building the audio-device helper"
    binary = helper_path()

    for attempt in range(30):
        try:
            if "BlackHole" in device_list():
                break
        except (OSError, subprocess.SubprocessError):
            pass
        if attempt == 0:
            yield ("waiting for BlackHole to appear; a fresh install may need "
                   "the audio daemon restarted")
        time.sleep(2)
    else:
        raise RuntimeError(
            "BlackHole never appeared. Install it with: "
            "brew install blackhole-2ch, then use 'Restart audio daemon' if "
            "this is a fresh Mac.")

    yield "creating the Aggregate Device"
    result = subprocess.run([str(binary), "ensure"], capture_output=True,
                            text=True, timeout=_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(f"device creation failed: {result.stderr.strip()}")
    subprocess.run([str(binary), "check"], capture_output=True, text=True,
                   timeout=_TIMEOUT, check=True)
    yield f"Aggregate Device '{device}' is ready"


def restart_daemon() -> None:
    """Restart coreaudiod through the standard macOS authorisation dialog.

    The UI has no TTY, so setup-mac.sh takes its no-sudo branch and skips this; a
    freshly installed BlackHole does not appear until coreaudiod restarts.

    The script string is a constant: anything interpolated between those quotes
    would be executed as shell by root. Nothing from a request reaches it.
    """
    subprocess.run(
        ["osascript", "-e",
         'do shell script "killall coreaudiod" with administrator privileges'],
        capture_output=True, text=True, timeout=120, check=True)
