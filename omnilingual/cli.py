"""Command-line entry point: omnilingual RECORDING [options]."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from omnilingual.audio.normalize import FfmpegError, FfmpegMissingError, ensure_ffmpeg
from omnilingual.cache import JsonCache
from omnilingual.config import ConfigError, load_settings
from omnilingual.http import AuthError, QuotaError, SarvamError
from omnilingual.models import Segment
from omnilingual.pipeline import estimate, prepare, run, work_dir_for
from omnilingual.render.markdown import fmt_ts, render, render_english_only
from omnilingual.stt.sarvam import SarvamSTT
from omnilingual.translate.mayura import MayuraTranslator

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _fail(msg: str, code: int = 1) -> None:
    console.print(f"[red]error:[/red] {msg}")
    raise typer.Exit(code)


@app.command()
def transcribe(
    recording: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Zoom local recording (.m4a/.mp4/any ffmpeg input)")],
    langs: Annotated[Optional[str], typer.Option(help="Comma-separated expected languages, e.g. hi-IN,ta-IN,en-IN. Used for warnings only.")] = None,
    out: Annotated[Optional[Path], typer.Option(help="Output markdown path. Default: <recording stem>.md next to input.")] = None,
    english_only: Annotated[bool, typer.Option("--english-only", help="Also write <stem>.en.md with English text only.")] = False,
    estimate_only: Annotated[bool, typer.Option("--estimate", help="Chunk and price the recording. No API calls.")] = False,
    api_key: Annotated[Optional[str], typer.Option(help="Sarvam API key. Overrides SARVAM_API_KEY.")] = None,
    work_dir: Annotated[Optional[Path], typer.Option(help="Cache/work root. Default: <out dir>/.omnilingual")] = None,
    max_chunk_s: Annotated[float, typer.Option(help="Max chunk length in seconds (<30).")] = 28.0,
    min_chunk_s: Annotated[float, typer.Option(help="Min chunk length in seconds.")] = 5.0,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING, format="%(levelname)s %(message)s")

    out_path = out or recording.with_suffix(".md")
    work_root = work_dir or out_path.parent / ".omnilingual"
    lang_list = [s.strip() for s in langs.split(",") if s.strip()] if langs else []
    settings = load_settings(api_key=api_key, langs=lang_list, max_chunk_s=max_chunk_s, min_chunk_s=min_chunk_s)

    try:
        ensure_ffmpeg()
    except FfmpegMissingError as exc:
        _fail(str(exc))

    wd = work_dir_for(recording, work_root)

    if estimate_only:
        try:
            duration, chunks = prepare(recording, wd, settings)
        except FfmpegError as exc:
            _fail(str(exc))
        cost = estimate(duration, chunks, settings)
        console.print(f"{recording.name}: {fmt_ts(duration)} audio, {len(chunks)} chunks")
        console.print(f"Projected: STT ₹{duration / 3600 * settings.stt_inr_per_hour:.2f} + MT ~₹{cost.mt_chars / 10_000 * settings.mt_inr_per_10k_chars:.2f} = ~₹{cost.inr_estimate:.2f}")
        raise typer.Exit(0)

    try:
        settings.require_key()
    except ConfigError as exc:
        _fail(str(exc))

    def progress(i: int, n: int, seg: Segment) -> None:
        mark = "" if seg.status == "ok" else f" [yellow]{seg.status}[/yellow]"
        console.print(f"[{i}/{n}] {fmt_ts(seg.chunk.start_s)} {seg.lang} {seg.prob:.2f}{mark}")

    try:
        transcript = run(recording, wd, settings, SarvamSTT(settings), MayuraTranslator(settings), JsonCache(wd / "cache"), progress)
    except FfmpegError as exc:
        _fail(str(exc))
    except QuotaError as exc:
        _fail(f"{exc}. Cached progress kept; re-run same command to resume.")
    except AuthError as exc:
        _fail(f"{exc}. Check your Sarvam API key.")
    except SarvamError as exc:
        _fail(str(exc))

    out_path.write_text(render(transcript), encoding="utf-8")
    console.print(f"Wrote {out_path}")
    if english_only:
        en_path = out_path.with_suffix(".en.md")
        en_path.write_text(render_english_only(transcript), encoding="utf-8")
        console.print(f"Wrote {en_path}")

    bad = sum(1 for s in transcript.segments if s.status != "ok")
    console.print(f"{len(transcript.segments)} segments · estimated cost ₹{transcript.cost.inr_estimate:.2f}")
    if bad:
        console.print(f"[yellow]{bad} segment(s) need attention (see notes in transcript).[/yellow]")
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
