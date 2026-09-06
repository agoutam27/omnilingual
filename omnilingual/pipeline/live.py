"""Live session pipeline: per-chunk STT/MT classification plus calibration."""

from __future__ import annotations

import logging

from omnilingual.cache import JsonCache
from omnilingual.config import Settings
from omnilingual.models import Chunk, Segment
from omnilingual.pipeline import (
    STT_FAILED_TEXT,
    _mt_cached,
    _price,
    _stt_cached,
)

log = logging.getLogger("omnilingual")


def calibrate_energy_floor(ambient_rms: float) -> float:
    """Speech floor from the 1 s ambient probe: 4x ambient, floor 0.004."""
    return max(ambient_rms * 4.0, DEFAULT_ENERGY_FLOOR)


def resolve_energy_floor(probe: bytes) -> tuple[float, bool]:
    """Floor for this run plus whether the probe was voice-contaminated.

    Speaking during the first second would otherwise set the floor at speech
    level and gate the whole meeting as silence — fall back to the default
    floor instead so speech is billed rather than missed.
    """
    if voice_band_share(probe) >= VOICE_MIN_SHARE:
        return DEFAULT_ENERGY_FLOOR, True
    return calibrate_energy_floor(rms(probe)), False


def process_chunk(
    chunk: Chunk,
    *,
    speech: bool,
    stt,
    translator,
    cache: JsonCache,
    settings: Settings,
) -> tuple[Segment, float, bool]:
    """Transcribe + translate one sealed chunk. Returns (segment, inr, billed).

    Mirrors the batch per-chunk classification exactly. QuotaError/AuthError
    propagate (the run loop converts them into a capture-while-halted state).
    """
    if not speech:
        seg = Segment(chunk=chunk, lang="", prob=0.0, text="",
                      english=None, status="no_speech")
        return (seg, 0.0, False)
    result = _stt_cached(chunk, stt, cache)
    if result is None:
        seg = Segment(chunk=chunk, lang="unknown", prob=0.0,
                      text=STT_FAILED_TEXT, english=None, status="stt_failed")
        return (seg, _price(chunk.duration_s, 0, settings), True)
    if settings.langs and result.lang not in settings.langs:
        log.warning(
            "chunk %d: detected %s (p=%.2f) outside configured languages %s",
            chunk.idx, result.lang, result.prob, ",".join(settings.langs),
        )
    if not result.text.strip():
        seg = Segment(chunk=chunk, lang=result.lang, prob=result.prob,
                      text="", english=None, status="no_speech")
        return (seg, _price(chunk.duration_s, 0, settings), True)
    if result.lang == "en-IN":
        seg = Segment(chunk=chunk, lang="en-IN", prob=result.prob,
                      text=result.text, english=None, status="ok")
        return (seg, _price(chunk.duration_s, 0, settings), True)
    if not translator.supports(result.lang):
        seg = Segment(chunk=chunk, lang=result.lang, prob=result.prob,
                      text=result.text, english=None, status="mt_unsupported")
        return (seg, _price(chunk.duration_s, 0, settings), True)
    english = _mt_cached(result.text, result.lang, translator, cache)
    if english is None:
        seg = Segment(chunk=chunk, lang=result.lang, prob=result.prob,
                      text=result.text, english=None, status="mt_failed")
        return (seg, _price(chunk.duration_s, 0, settings), True)
    seg = Segment(chunk=chunk, lang=result.lang, prob=result.prob,
                  text=result.text, english=english, status="ok")
    return (seg, _price(chunk.duration_s, len(result.text), settings), True)


"""(continued) Threaded live run loop."""

import json
import os
import queue
import select
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from omnilingual.audio.live_capture import (
    BYTES_PER_SECOND,
    VOICE_MIN_SHARE,
    CaptureError,
    LiveCapture,
    parse_silence_line,
    rms,
    voice_band_share,
)
from omnilingual.audio.live_slicer import DEFAULT_ENERGY_FLOOR, LiveSlicer
from omnilingual.http import AuthError, QuotaError
from omnilingual.pipeline import LIVE_SESSION_KIND
from omnilingual.render.live import LiveEnglishWriter, LiveMarkdownWriter
from omnilingual.render.markdown import fmt_ts


@dataclass
class LiveOptions:
    out: Path
    device: str = "Omnilingual"
    mic_only: bool = False
    target_s: float = 8.0
    max_chunk_s: float = 28.0
    min_chunk_s: float = 5.0
    noise_db: float = -35.0
    stt_workers: int = 2
    max_cost: float = 50.0
    work_root: Path | None = None
    english_only: bool = False


_HALT_TEXT = {
    "cost": "[transcription stopped: cost cap reached]",
    "quota": "[transcription stopped: API quota exceeded]",
    "auth": "[transcription stopped: authorization failed]",
}


def run_live(opts: LiveOptions, settings, stt, translator, *,
             status: Callable[[str], None] | None = None,
             capture_factory: Callable = LiveCapture) -> int:
    """Run a live session. Returns the process exit code (0/1/2)."""
    say = status or (lambda msg: None)
    root = opts.work_root or opts.out.parent / ".omnilingual"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    session = root / f"live-{stamp}"
    (session / "live-chunks").mkdir(parents=True, exist_ok=True)
    (session / "session.json").write_text(json.dumps({
        "kind": LIVE_SESSION_KIND,
        "device": opts.device,
        "mic_only": opts.mic_only,
        "langs": list(settings.langs),
        "target_s": opts.target_s,
        "max_chunk_s": opts.max_chunk_s,
        "min_chunk_s": opts.min_chunk_s,
        "started_utc": stamp,
    }, indent=2), encoding="utf-8")
    cache = JsonCache(session / "cache")

    capture = capture_factory(opts.device, mic_only=opts.mic_only,
                              noise_db=opts.noise_db)
    try:
        capture.open()
    except CaptureError as exc:
        say(f"[live] capture failed: {exc}")
        return 1

    probe = capture.read(BYTES_PER_SECOND)  # 1 s ambient probe, then reused
    energy_floor, contaminated = resolve_energy_floor(probe)
    if contaminated:
        say("[live] warning: you were already speaking during the 1 s ambient "
            "probe; using the default energy floor. Restart quiet for a "
            "tighter floor.")
    slicer = LiveSlicer(session, target_s=opts.target_s,
                        max_s=opts.max_chunk_s, min_s=opts.min_chunk_s,
                        energy_floor=energy_floor)

    title = "Meeting transcript — LIVE " + datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M %Z")
    main_w = LiveMarkdownWriter(opts.out, title=title, cost_cap=opts.max_cost)
    en_w = (LiveEnglishWriter(opts.out.with_suffix(".en.md"), title=title,
                              cost_cap=opts.max_cost)
            if opts.english_only else None)
    writers = [w for w in (main_w, en_w) if w is not None]

    jobs: queue.Queue = queue.Queue()
    results: dict[int, tuple[Segment, float, bool]] = {}
    cond = threading.Condition()
    mlock = threading.Lock()
    slicer_lock = threading.Lock()
    gap_box: list[tuple[float, float]] = []
    state = {"accrued": 0.0, "halt": None, "enqueued": 0, "next": 0,
             "last_kept_end": 0.0, "sealed_end": 0.0, "sealed_n": 0,
             "appended_n": 0, "appended_end": 0.0, "bad": False, "died": False}
    stop = threading.Event()
    feeding_done = threading.Event()
    capture_done = threading.Event()
    t0 = time.monotonic()

    def elapsed() -> str:
        return fmt_ts(time.monotonic() - t0)

    def emit(sealed_list) -> None:
        with mlock:
            for sc in sealed_list:
                jobs.put((state["enqueued"], sc))
                state["enqueued"] += 1
                state["sealed_n"] += 1
                state["sealed_end"] = max(state["sealed_end"], sc.chunk.end_s)
        with cond:
            cond.notify_all()

    def halt_once(kind: str, msg: str) -> None:
        with mlock:
            if state["halt"] is None:
                state["halt"] = kind
                say(msg)

    def capture_loop() -> None:
        # Reads stay explicit: __iter__ reports a dead child as a clean end,
        # and only an explicit read surfaces CaptureError, so collapsing
        # this loop into `for block in capture` would silently break
        # exit-2-on-death. Scripted captures end dry with StopIteration.
        try:
            with slicer_lock:
                emit(slicer.feed(probe))
            while not stop.is_set():
                try:
                    block = capture.read(BYTES_PER_SECOND)
                except StopIteration:
                    break
                with slicer_lock:
                    sealed = []
                    for start, end in gap_box:
                        sealed += slicer.note_gap(start, end)
                    del gap_box[:]
                    sealed += slicer.feed(block)
                emit(sealed)
        except CaptureError as exc:
            with mlock:
                state["died"] = True
            say(f"[live {elapsed()}] capture died: {exc}")
        finally:
            feeding_done.set()
            capture.close()

    def stderr_loop() -> None:
        raw = capture.stderr
        if raw is None:
            return
        pending_start: float | None = None
        if hasattr(raw, "readline"):  # real pipe: poll until feeding ends
            while True:
                ready, _, _ = select.select([raw], [], [], 0.2)
                if not ready:
                    if feeding_done.is_set():
                        return
                    continue
                line = raw.readline()
                if not line:
                    return
                text = line.decode("utf-8", "replace") if isinstance(
                    line, bytes) else line
                parsed = parse_silence_line(text)
                if parsed is None:
                    continue
                kind, ts = parsed
                if kind == "start":
                    pending_start = ts
                elif pending_start is not None:
                    with slicer_lock:
                        gap_box.append((pending_start, ts))
                    pending_start = None
        else:  # scripted iterable (tests): drain fast
            for raw_line in raw:
                text = raw_line.decode("utf-8", "replace") if isinstance(
                    raw_line, bytes) else raw_line
                parsed = parse_silence_line(text)
                if parsed is None:
                    continue
                kind, ts = parsed
                if kind == "start":
                    pending_start = ts
                elif pending_start is not None:
                    with slicer_lock:
                        gap_box.append((pending_start, ts))
                    pending_start = None

    def worker() -> None:
        while True:
            job = jobs.get()
            if job is None:
                return
            seq, sc = job
            with mlock:
                halt = state["halt"]
                over = state["accrued"] >= opts.max_cost
            if halt is not None or over:
                if halt is None:
                    halt_once("cost",
                              f"[live {elapsed()}] cost cap ₹{opts.max_cost:.0f} "
                              "reached; capturing continues, "
                              "run --from-chunks to recover")
                    halt = "cost"
                seg = Segment(chunk=sc.chunk, lang="unknown", prob=0.0,
                              text=_HALT_TEXT[halt], english=None,
                              status="stt_failed")
                item = (seg, 0.0, False)
            else:
                try:
                    item = process_chunk(sc.chunk, speech=sc.speech, stt=stt,
                                         translator=translator, cache=cache,
                                         settings=settings)
                except QuotaError:
                    halt_once("quota",
                              f"[live {elapsed()}] API quota exceeded (402); "
                              "capturing continues, run --from-chunks to recover")
                    seg = Segment(chunk=sc.chunk, lang="unknown", prob=0.0,
                                  text=_HALT_TEXT["quota"], english=None,
                                  status="stt_failed")
                    item = (seg, 0.0, False)
                except AuthError:
                    halt_once("auth",
                              f"[live {elapsed()}] authorization failed; "
                              "capturing continues")
                    seg = Segment(chunk=sc.chunk, lang="unknown", prob=0.0,
                                  text=_HALT_TEXT["auth"], english=None,
                                  status="stt_failed")
                    item = (seg, 0.0, False)
            with cond:
                results[seq] = item
                cond.notify_all()

    def status_line(seq: int, seg: Segment) -> str:
        dur = seg.chunk.duration_s
        idx = seg.chunk.idx if seg.chunk.idx >= 0 else seq
        if seg.status == "ok" and seg.lang != "en-IN":
            tail = f"→ {seg.lang} {seg.prob:.2f} → en ✓"
        elif seg.status == "ok":
            tail = f"→ {seg.lang} {seg.prob:.2f} ✓"
        elif seg.status == "no_speech":
            tail = "→ silence"
        else:
            tail = f"→ {seg.status}"
        return f"[live {elapsed()}] sealed #{idx} ({dur:.1f} s) {tail}"

    threads = [threading.Thread(target=capture_loop, daemon=True),
               threading.Thread(target=stderr_loop, daemon=True)]
    workers = [threading.Thread(target=worker, daemon=True)
               for _ in range(max(1, opts.stt_workers))]
    for th in threads + workers:
        th.start()

    # Two-stage Ctrl+C: first stops gracefully, second reaps ffmpeg and exits.
    sigints = 0
    prev = signal.getsignal(signal.SIGINT)

    def on_sigint(signum, frame):
        nonlocal sigints
        sigints += 1
        if sigints == 1:
            say(f"[live {elapsed()}] stopping… (Ctrl+C again to quit now)")
            stop.set()
        else:
            capture.close()
            os._exit(2)

    signal.signal(signal.SIGINT, on_sigint)
    last_lag_log = t0
    final = False
    try:
        while True:
            with cond:
                have_item = False
                while (state["next"] not in results
                       and not (feeding_done.is_set()
                                and state["next"] >= state["enqueued"])):
                    cond.wait(timeout=0.5)
                if state["next"] in results:
                    item = results.pop(state["next"])
                    state["next"] += 1
                    have_item = True
                elif feeding_done.is_set() and not final:
                    # Drain gaps the stderr thread parsed after feeding ended.
                    # The tail jobs emitted here are appended through the
                    # normal ordered path above — never dropped.
                    pending = None
                    with slicer_lock:
                        if gap_box:
                            pending = list(gap_box)
                            del gap_box[:]
                    with slicer_lock:
                        tail = []
                        if pending:
                            for start, end in pending:
                                tail += slicer.note_gap(start, end)
                        tail += slicer.flush()
                    emit(tail)
                    for _ in workers:
                        jobs.put(None)
                    final = True
                elif (feeding_done.is_set() and final
                        and state["next"] >= state["enqueued"]):
                    for th in threads + workers:
                        th.join(timeout=30)
                    break
                else:
                    continue
            if not have_item:
                continue
            seg, delta, _ = item
            say(status_line(state["next"] - 1, seg))
            keep = True
            if (seg.status == "no_speech"
                    and seg.chunk.end_s - state["last_kept_end"] <= 30.0):
                keep = False
            if keep:
                for w in writers:
                    w.append_segment(seg, delta)
                with mlock:
                    state["accrued"] += delta
                    state["appended_n"] += 1
                    state["appended_end"] = seg.chunk.end_s
                    state["last_kept_end"] = seg.chunk.end_s
                    if seg.status != "ok":
                        state["bad"] = True
            now = time.monotonic()
            if now - last_lag_log >= 30.0:
                last_lag_log = now
                with mlock:
                    lag = state["sealed_end"] - state["appended_end"]
                    say(f"[live {elapsed()}] sealed {state['sealed_n']} · "
                        f"appended {state['appended_n']} · lag ~{lag:.0f} s")
    finally:
        signal.signal(signal.SIGINT, prev)
        for w in writers:
            try:
                w.finalize()
            finally:
                w.close()
    if state["died"]:
        say(f"[live {elapsed()}] session ended early (capture died); "
            "file remains valid")
    if state["halt"] == "cost":
        say(f"[live {elapsed()}] stopped API calls at the ₹{opts.max_cost:.0f} cap; "
            "sealed chunks remain on disk for --from-chunks recovery")
    return 2 if (state["bad"] or state["died"]) else 0
