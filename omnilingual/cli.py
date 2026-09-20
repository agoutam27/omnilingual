"""Command-line entry point: omnilingual RECORDING [options]."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Annotated, NoReturn, Optional

import typer
from rich.console import Console
from rich.markup import escape

from omnilingual.audio.live_capture import (
    BYTES_PER_SECOND,
    CaptureError,
    LiveCapture,
    dbfs,
    parse_devices,
    rms,
    voice_band_share,
    VOICE_MIN_SHARE,
)
from omnilingual.audio.normalize import FfmpegError, FfmpegMissingError, ensure_ffmpeg
from omnilingual.cache import JsonCache
from omnilingual.config import ConfigError, load_settings
from omnilingual.http import AuthError, QuotaError, SarvamError
from omnilingual.models import Segment, chunks_from_json
from omnilingual.pipeline import LIVE_SESSION_KIND, estimate, prepare, run, run_from_chunks, work_dir_for
from omnilingual.pipeline.live import LiveOptions, run_live
from omnilingual.render.markdown import fmt_ts, render, render_english_only
from omnilingual.stt import build_stt
from omnilingual.translate.mayura import MayuraTranslator

class _AppGroup(typer.core.TyperGroup):
    """Routes bare `omnilingual RECORDING …` (v1 UX) to `transcribe`.

    Click groups send every token after the first positional to the
    subcommand, so trailing flags (`recording --api-key k`) cannot parse on
    a group. Rewriting up front keeps one option parser (transcribe's own,
    which accepts flags anywhere) instead of two signatures to keep in sync.
    """

    def parse_args(self, ctx, args):
        if args and args[0] not in self.commands and args[0] not in ("-h", "--help"):
            if not args[0].startswith("-"):
                args = ["transcribe", *args]
        return super().parse_args(ctx, args)


app = typer.Typer(add_completion=False, no_args_is_help=True, cls=_AppGroup)
console = Console()


# Sarvam's synchronous speech-to-text endpoint rejects audio of 30 seconds or more.
MAX_CHUNK_LIMIT_S = 30.0


def _fail(msg: str, code: int = 1) -> NoReturn:
    # msg carries ffmpeg stderr and API response bodies, which are full of square
    # brackets that Rich would otherwise parse as markup tags: at best the text is
    # silently swallowed, at worst rendering raises.
    console.print(f"[red]error:[/red] {escape(msg)}")
    raise typer.Exit(code)


def _run_from_chunks(session_dir: Path, *, out: Path | None,
                     english_only: bool, estimate_only: bool,
                     api_key: str | None, work_dir: Path | None,
                     stt_provider: str, stt_model: str | None) -> None:
    """Finish a halted live session. Never returns (raises typer.Exit)."""
    if work_dir is not None:
        _fail("--work-dir has no effect with --from-chunks (the session cache is reused)")
    try:
        if json.loads((session_dir / "session.json").read_text(
                encoding="utf-8")).get("kind") != LIVE_SESSION_KIND:
            raise ValueError(f"kind is not {LIVE_SESSION_KIND!r}")
        chunks = chunks_from_json((session_dir / "chunks.json").read_text(
            encoding="utf-8"))
        if not chunks:
            raise ValueError("live session has no sealed chunks yet")
    except (OSError, ValueError) as exc:
        _fail(f"{session_dir} is not a live-session dir ({exc})")
    assert chunks  # narrowed by the guard above
    try:
        settings = load_settings(api_key=api_key, stt_provider=stt_provider, stt_model=stt_model)
        stt = build_stt(settings)
        settings.require_key()
        if settings.stt_provider == "groq":
            settings.require_groq_key()
    except ConfigError as exc:
        _fail(str(exc))
    if estimate_only:
        duration = sum(c.duration_s for c in chunks)
        cost = estimate(duration, chunks, settings, stt=stt)
        console.print(f"{escape(session_dir.name)}: {fmt_ts(duration)} audio, {len(chunks)} chunks")
        console.print(f"Projected: STT ₹{duration / 3600 * getattr(stt, 'inr_per_hour', settings.stt_inr_per_hour):.2f} + MT ~₹{cost.mt_chars / 10_000 * settings.mt_inr_per_10k_chars:.2f} = ~₹{cost.inr_estimate:.2f}")
        raise typer.Exit(0)
    out_path = out or (session_dir / "transcript.md")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _fail(f"cannot create output directory {out_path.parent}: {exc}")

    def progress(i: int, n: int, seg: Segment) -> None:
        mark = "" if seg.status == "ok" else f" [yellow]{seg.status}[/yellow]"
        # seg.lang is whatever the API reported, so it is escaped like any other
        # external value; the counter and the mark are ours.
        console.print(f"\\[{i}/{n}] {fmt_ts(seg.chunk.start_s)} {escape(seg.lang)} {seg.prob:.2f}{mark}")

    transcript = run_from_chunks(session_dir, settings, stt,
                                 MayuraTranslator(settings),
                                 JsonCache(session_dir / "cache"), progress)
    out_path.write_text(render(transcript), encoding="utf-8")
    console.print(f"Wrote {escape(str(out_path))}")
    if english_only:
        en_path = out_path.with_suffix(".en.md")
        en_path.write_text(render_english_only(transcript), encoding="utf-8")
        console.print(f"Wrote {escape(str(en_path))}")

    bad = sum(1 for s in transcript.segments if s.status != "ok")
    console.print(f"{len(transcript.segments)} segments · estimated cost ₹{transcript.cost.inr_estimate:.2f}")
    if bad:
        console.print(f"[yellow]{bad} segment(s) need attention (see notes in transcript).[/yellow]")
        raise typer.Exit(2)
    raise typer.Exit(0)


@app.command()
def transcribe(
    recording: Annotated[Optional[Path], typer.Argument(exists=True, dir_okay=False, readable=True, help="Zoom local recording (.m4a/.mp4/any ffmpeg input)")] = None,
    langs: Annotated[Optional[str], typer.Option(help="Comma-separated expected languages, e.g. hi-IN,ta-IN,en-IN. Used for warnings only.")] = None,
    out: Annotated[Optional[Path], typer.Option(help="Output markdown path. Default: <recording stem>.md next to input.")] = None,
    english_only: Annotated[bool, typer.Option("--english-only", help="Also write <stem>.en.md with English text only.")] = False,
    estimate_only: Annotated[bool, typer.Option("--estimate", help="Chunk and price the recording. No API calls.")] = False,
    api_key: Annotated[Optional[str], typer.Option(help="Sarvam API key. Overrides SARVAM_API_KEY.")] = None,
    stt_provider: Annotated[str, typer.Option("--stt", help="Speech-to-text backend: sarvam (default), mlx-whisper (Apple Silicon, free), faster-whisper (Intel/Linux CPU, free), groq (cloud, cheap).")] = "sarvam",
    stt_model: Annotated[Optional[str], typer.Option("--stt-model", help="Override the STT model id for the chosen provider.")] = None,
    work_dir: Annotated[Optional[Path], typer.Option(help="Cache/work root. Default: <out dir>/.omnilingual")] = None,
    from_chunks: Annotated[Optional[Path], typer.Option("--from-chunks", help="Finish a halted live session dir")] = None,
    max_chunk_s: Annotated[float, typer.Option(help="Max chunk length in seconds (<30).")] = 28.0,
    min_chunk_s: Annotated[float, typer.Option(help="Min chunk length in seconds.")] = 5.0,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING, format="%(levelname)s %(message)s")

    if from_chunks is not None and recording is not None:
        _fail("--from-chunks cannot be combined with RECORDING")
    if from_chunks is not None:
        _run_from_chunks(from_chunks, out=out, english_only=english_only,
                          estimate_only=estimate_only, api_key=api_key, work_dir=work_dir,
                          stt_provider=stt_provider, stt_model=stt_model)
    if recording is None:
        _fail("Missing argument 'RECORDING'.")

    out_path = out or recording.with_suffix(".md")
    work_root = work_dir or out_path.parent / ".omnilingual"
    lang_list = [s.strip() for s in langs.split(",") if s.strip()] if langs else []
    try:
        settings = load_settings(api_key=api_key, langs=lang_list, max_chunk_s=max_chunk_s,
                                 min_chunk_s=min_chunk_s, stt_provider=stt_provider, stt_model=stt_model)
        stt = build_stt(settings)
    except ConfigError as exc:
        _fail(str(exc))

    # Bad chunk bounds would only surface after normalizing the whole recording, or
    # worse, as a wall of 400s from the API. Check them before doing any work.
    if not 0 < min_chunk_s < max_chunk_s < MAX_CHUNK_LIMIT_S or max_chunk_s < 2 * min_chunk_s:
        _fail("--max-chunk-s must be < 30 and > --min-chunk-s, and at least 2x --min-chunk-s")

    # An unwritable output path must not be discovered after paying for transcription.
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _fail(f"cannot create output directory {out_path.parent}: {exc}")

    try:
        ensure_ffmpeg()
    except FfmpegMissingError as exc:
        _fail(str(exc))

    wd = work_dir_for(recording, work_root)

    if estimate_only:
        try:
            duration, chunks = prepare(recording, wd, settings)
        except (FfmpegError, FfmpegMissingError) as exc:
            _fail(str(exc))
        cost = estimate(duration, chunks, settings, stt=stt)
        console.print(f"{escape(recording.name)}: {fmt_ts(duration)} audio, {len(chunks)} chunks")
        console.print(f"Projected: STT ₹{duration / 3600 * getattr(stt, 'inr_per_hour', settings.stt_inr_per_hour):.2f} + MT ~₹{cost.mt_chars / 10_000 * settings.mt_inr_per_10k_chars:.2f} = ~₹{cost.inr_estimate:.2f}")
        raise typer.Exit(0)

    try:
        settings.require_key()
        if settings.stt_provider == "groq":
            settings.require_groq_key()
    except ConfigError as exc:
        _fail(str(exc))

    def progress(i: int, n: int, seg: Segment) -> None:
        mark = "" if seg.status == "ok" else f" [yellow]{seg.status}[/yellow]"
        # seg.lang is whatever the API reported, so it is escaped like any other
        # external value; the counter and the mark are ours.
        console.print(f"\\[{i}/{n}] {fmt_ts(seg.chunk.start_s)} {escape(seg.lang)} {seg.prob:.2f}{mark}")

    try:
        transcript = run(recording, wd, settings, stt, MayuraTranslator(settings), JsonCache(wd / "cache"), progress)
    except (FfmpegError, FfmpegMissingError) as exc:
        _fail(str(exc))
    except QuotaError as exc:
        _fail(f"{exc}. Cached progress kept; re-run same command to resume.")
    except AuthError as exc:
        _fail(f"{exc}. Check your API key.")
    except SarvamError as exc:
        _fail(str(exc))

    out_path.write_text(render(transcript), encoding="utf-8")
    console.print(f"Wrote {escape(str(out_path))}")
    if english_only:
        en_path = out_path.with_suffix(".en.md")
        en_path.write_text(render_english_only(transcript), encoding="utf-8")
        console.print(f"Wrote {escape(str(en_path))}")

    bad = sum(1 for s in transcript.segments if s.status != "ok")
    console.print(f"{len(transcript.segments)} segments · estimated cost ₹{transcript.cost.inr_estimate:.2f}")
    if bad:
        console.print(f"[yellow]{bad} segment(s) need attention (see notes in transcript).[/yellow]")
        raise typer.Exit(2)


@app.command()
def live(
    out: Annotated[Optional[Path], typer.Option("--out")] = None,
    input: Annotated[str, typer.Option("--input")] = "Omnilingual",
    mic_only: Annotated[bool, typer.Option("--mic-only")] = False,
    langs: Annotated[str, typer.Option("--langs")] = "",
    target_s: Annotated[float, typer.Option("--target-s")] = 8.0,
    max_chunk_s: Annotated[float, typer.Option("--max-chunk-s")] = 28.0,
    min_chunk_s: Annotated[float, typer.Option("--min-chunk-s")] = 5.0,
    noise_db: Annotated[float, typer.Option("--noise-db")] = -35.0,
    stt_workers: Annotated[int, typer.Option("--stt-workers")] = 2,
    max_cost: Annotated[float, typer.Option("--max-cost")] = 50.0,
    check_audio: Annotated[bool, typer.Option("--check-audio")] = False,
    english_only: Annotated[bool, typer.Option("--english-only")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
    stt_provider: Annotated[str, typer.Option("--stt", help="Speech-to-text backend: sarvam (default), mlx-whisper (Apple Silicon, free), faster-whisper (Intel/Linux CPU, free), groq (cloud, cheap).")] = "sarvam",
    stt_model: Annotated[str | None, typer.Option("--stt-model", help="Override the STT model id for the chosen provider.")] = None,
    work_dir: Annotated[Path | None, typer.Option("--work-dir")] = None,
    verbose: Annotated[bool, typer.Option("-v")] = False,
):
    """Transcribe mic + system audio live into a growing Markdown file."""
    err = Console(stderr=True)
    say = lambda m: err.print(m, markup=False)  # brackets must not render
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)
    if not 0 < min_chunk_s < max_chunk_s < MAX_CHUNK_LIMIT_S or max_chunk_s < 2 * min_chunk_s:
        _fail("--max-chunk-s must be < 30 and > --min-chunk-s, and at least 2x --min-chunk-s")
    if not min_chunk_s <= target_s <= max_chunk_s:
        _fail("--target-s must be between --min-chunk-s and --max-chunk-s")
    if stt_workers < 1:
        _fail("--stt-workers must be >= 1")
    try:
        ensure_ffmpeg()
    except FfmpegMissingError as exc:
        _fail(str(exc))
    if check_audio:
        listing = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            capture_output=True, text=True)
        devices = parse_devices((listing.stderr or "") + (listing.stdout or ""))
        if not devices:
            _fail("No AVFoundation audio devices found.")
        for d in devices:
            say(f"[{d.index}] {d.name}")
        cap = LiveCapture(input, mic_only=mic_only, noise_db=noise_db)
        try:
            cap.open()
            with cap:
                probe = cap.read(BYTES_PER_SECOND)
        except CaptureError as exc:
            _fail(str(exc))
        level = dbfs(rms(probe))
        say(f"input '{input}': 1 s probe {level:.1f} dBFS")
        if level < -50:
            say("near silence: set the system output to the Multi-Output "
                "Device (headphones + BlackHole) and confirm the Aggregate "
                "Device 'Omnilingual' contains mic + BlackHole")
        elif voice_band_share(probe) >= VOICE_MIN_SHARE:
            say("voice-band check: VOICE LIKELY — the mic is delivering speech frequencies")
        else:
            say("voice-band check: NO VOICE — only low-frequency energy "
                "(mains hum?); check the Aggregate mic selection, gain, and mute")
        return
    if out is None:
        _fail("Missing option '--out'.")
    out_path = out
    if out_path.exists():
        stamp = datetime.now().strftime("%H%M")
        cand = out_path.with_name(f"{out_path.stem}-{stamp}{out_path.suffix}")
        n = 2
        while cand.exists():
            cand = out_path.with_name(
                f"{out_path.stem}-{stamp}-{n}{out_path.suffix}")
            n += 1
        say(f"--out exists; writing {cand.name} instead")
        out_path = cand
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lang_list = [s.strip() for s in langs.split(",") if s.strip()] if langs else []
    try:
        settings = load_settings(api_key=api_key, langs=lang_list,
                                 stt_provider=stt_provider, stt_model=stt_model)
        stt = build_stt(settings)
        settings.require_key()
        if settings.stt_provider == "groq":
            settings.require_groq_key()
    except ConfigError as exc:
        _fail(str(exc))
    opts = LiveOptions(
        out=out_path, device=input, mic_only=mic_only,
        target_s=target_s, max_chunk_s=max_chunk_s, min_chunk_s=min_chunk_s,
        noise_db=noise_db, stt_workers=stt_workers, max_cost=max_cost,
        work_root=work_dir, english_only=english_only)
    code = run_live(opts, settings, stt,
                    MayuraTranslator(settings), status=say)
    raise typer.Exit(code=code)


if __name__ == "__main__":
    app()
