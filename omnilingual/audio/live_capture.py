"""Live audio capture: one ffmpeg child reads a macOS Aggregate device.

Stdout carries 16 kHz mono s16le PCM; stderr carries silencedetect events.
Exactly one ``-i`` input is used (dual-input amix is forbidden).
"""

from __future__ import annotations

import array
import math
import re
import subprocess
from dataclasses import dataclass
from typing import Iterator

from .chunker import SILENCE_END_RE, SILENCE_START_RE
from .normalize import ensure_ffmpeg

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # mono s16le
_READ_BLOCK = BYTES_PER_SECOND  # 1 s per iteration
_STARTUP_GRACE_S = 0.5

_DEVICE_LINE = re.compile(r"\[(\d+)\]\s+(.+?)\s*$")


class CaptureError(RuntimeError):
    """Raised when the capture child cannot start or dies mid-session."""


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str


def parse_devices(output: str) -> list[AudioDevice]:
    """Parse `ffmpeg -f avfoundation -list_devices true -i ""` output.

    Only the audio-device section is returned; video devices are ignored.
    """
    _, _, audio = output.partition("AVFoundation audio devices:")
    devices: list[AudioDevice] = []
    for line in audio.splitlines():
        m = _DEVICE_LINE.search(line)
        if m:
            devices.append(AudioDevice(index=int(m.group(1)), name=m.group(2)))
    return devices


def resolve_device(devices: list[AudioDevice], query: str) -> AudioDevice:
    """Resolve an unambiguous case-insensitive substring device name."""
    hits = [d for d in devices if query.lower() in d.name.lower()]
    if len(hits) == 1:
        return hits[0]
    names = ", ".join(f"[{d.index}] {d.name}" for d in devices) or "(none found)"
    if not hits:
        raise CaptureError(
            f"Audio device '{query}' not found. Available: {names}. "
            "See the setup steps: create the 'Omnilingual' Aggregate Device, "
            "then run `omnilingual live --check-audio`."
        )
    dupes = ", ".join(f"[{d.index}] {d.name}" for d in hits)
    raise CaptureError(
        f"Audio device name '{query}' is ambiguous: {dupes}. "
        "Rename the Aggregate Device to exactly 'Omnilingual'."
    )


def downmix_filter(mic_only: bool) -> str:
    """Downmix to 16 kHz mono.

    The Aggregate Device holds BlackHole (2 ch) first and the mic (1 ch) last,
    mixed with the voice channel dominant. Mic-only mode selects the mic
    channel (c2) so the output is pure mic at full scale.
    A channel-layout mismatch fails fast inside open() with a fix-it hint.
    """
    if mic_only:
        return "pan=mono|c0=c2,aresample=16000"
    return "pan=mono|c0=0.5*c0+0.25*c1+0.25*c2,aresample=16000"


def build_command(device_index: int, mic_only: bool, noise_db: float) -> list[str]:
    chain = (
        f"asplit=2[pcm][det];[pcm]{downmix_filter(mic_only)}[out];"
        f"[det]silencedetect=noise={noise_db}dB:d=0.4,anullsink"
    )
    return [
        "ffmpeg", "-hide_banner", "-nostdin",
        "-f", "avfoundation", "-thread_queue_size", "8",
        "-i", f":{device_index}",
        "-filter_complex", chain,
        "-map", "[out]", "-c:a", "pcm_s16le", "-f", "s16le", "-",
    ]


def parse_silence_line(line: str) -> tuple[str, float] | None:
    m = SILENCE_START_RE.search(line)
    if m:
        return ("start", float(m.group(1)))
    m = SILENCE_END_RE.search(line)
    if m:
        return ("end", float(m.group(1)))
    return None


def rms(samples: bytes) -> float:
    """RMS energy of s16le PCM, 0.0..1.0. Uses array/struct (audioop is banned)."""
    if not samples:
        return 0.0
    vals = array.array("h")
    vals.frombytes(samples)
    return math.sqrt(sum(v * v for v in vals) / len(vals)) / 32768


def dbfs(v: float) -> float:
    return 20 * math.log10(v) if v > 0 else float("-inf")


# First-difference energy ratio of a pure tone at f Hz sampled at 16 kHz is
# about (2*pi*f/16000)^2: ~0.0015 at 100 Hz mains hum, ~0.03 at 440 Hz speech,
# higher for broadband voice. 0.02 sits between hum harmonics and voice.
VOICE_MIN_SHARE = 0.02


def voice_band_share(samples: bytes) -> float:
    """Fraction of energy above ~300 Hz, via a first-difference highpass.

    stdlib has no FFT; the first difference attenuates low frequencies
    quadratically, which is exactly the hum-vs-voice split --check-audio
    needs. Returns 0.0 for silence. Uses array (audioop is banned)."""
    if not samples:
        return 0.0
    vals = array.array("h")
    vals.frombytes(samples)
    sig = sum(v * v for v in vals)
    if sig == 0:
        return 0.0
    diff = sum((b - a) * (b - a) for a, b in zip(vals, vals[1:]))
    return diff / sig


def read_exact(stream, n: int) -> bytes:
    """Read exactly n bytes or raise CaptureError on a truncated stream."""
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        piece = stream.read(remaining)
        if not piece:
            raise CaptureError(
                f"Audio stream ended {remaining} bytes early; "
                "the capture child likely died."
            )
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)


def describe_startup_failure(device_name: str, stderr_tail: str) -> str:
    low = stderr_tail.lower()
    if "not permitted" in low or "permission" in low:
        return (
            f"Could not open audio device '{device_name}': microphone access denied. "
            "Open System Settings ▸ Privacy & Security ▸ Microphone, enable "
            "your terminal app, then re-run."
        )
    if "no such" in low or "not found" in low or "invalid" in low:
        return (
            f"Could not open audio device '{device_name}': device not found. "
            "Create the 'Omnilingual' Aggregate Device (mic + BlackHole), "
            "then run `omnilingual live --check-audio`."
        )
    return (
        f"Could not open audio device '{device_name}': {stderr_tail.strip()[-500:]} "
        "Check the Aggregate Device channel layout (mic + 2ch BlackHole) "
        "and run `omnilingual live --check-audio`."
    )


class LiveCapture:
    """An ffmpeg avfoundation child yielding gapless 16 kHz mono s16le PCM."""

    def __init__(self, device_name: str, *, mic_only: bool = False,
                 noise_db: float = -35.0) -> None:
        self.device_name = device_name
        self.mic_only = mic_only
        self.noise_db = noise_db
        self._proc: subprocess.Popen | None = None

    def open(self) -> None:
        ensure_ffmpeg()
        listing = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            capture_output=True, text=True,
        )
        device = resolve_device(
            parse_devices((listing.stderr or "") + (listing.stdout or "")),
            self.device_name,
        )
        proc = subprocess.Popen(
            build_command(device.index, self.mic_only, self.noise_db),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
        )
        try:
            proc.wait(timeout=_STARTUP_GRACE_S)
        except subprocess.TimeoutExpired:
            self._proc = proc  # healthy: still running
            return
        tail = (proc.stderr.read() or b"").decode("utf-8", "replace")[-2000:]
        raise CaptureError(describe_startup_failure(self.device_name, tail))

    def read(self, n: int) -> bytes:
        if self._proc is None or self._proc.stdout is None:
            raise CaptureError("Capture is not open; call open() first.")
        try:
            return read_exact(self._proc.stdout, n)
        except CaptureError as exc:
            raise CaptureError(f"{exc} (device '{self.device_name}')") from exc

    def __iter__(self) -> Iterator[bytes]:
        while True:
            try:
                yield self.read(_READ_BLOCK)
            except CaptureError:
                return

    @property
    def stderr(self):
        return None if self._proc is None else self._proc.stderr

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        finally:
            for pipe in (proc.stdout, proc.stderr):
                try:
                    if pipe is not None:
                        pipe.close()
                except (BrokenPipeError, ValueError):
                    pass

    def __enter__(self) -> LiveCapture:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
