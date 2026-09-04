"""Convert any ffmpeg-readable recording to 16 kHz mono 16-bit PCM WAV."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FfmpegMissingError(RuntimeError):
    pass


class FfmpegError(RuntimeError):
    pass


def ensure_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise FfmpegMissingError(
            f"{', '.join(missing)} not found on PATH. Install with: brew install ffmpeg"
        )


def run_tool(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg/ffprobe command, surfacing stderr on failure.

    Raises FfmpegMissingError if binary is not found.
    Raises FfmpegError if command exits non-zero.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise FfmpegMissingError(
            f"{cmd[0]} not found on PATH. Install with: brew install ffmpeg"
        ) from e

    if result.returncode != 0:
        stderr_msg = result.stderr.strip()[-2000:]
        raise FfmpegError(
            f"{cmd[0]} failed (exit {result.returncode}): {stderr_msg}"
        )

    return result


def probe_duration(path: Path) -> float:
    result = run_tool(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
    )
    return float(result.stdout.strip())


def normalize(src: Path, dst: Path) -> float:
    """Write dst atomically: a failed or interrupted ffmpeg must not leave a half-WAV
    that a later resumed run would mistake for a complete normalization."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    try:
        run_tool(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(src),
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                # ffmpeg infers the container from the output extension, which is
                # ".part" here, so state it explicitly.
                "-f", "wav",
                str(part),
            ],
        )
    except FfmpegError:
        part.unlink(missing_ok=True)
        raise
    part.replace(dst)
    return probe_duration(dst)
