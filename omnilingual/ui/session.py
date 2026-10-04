"""One run, one worker thread, one stream of event dicts.

This is the ONLY module that knows how a run executes. The HTTP layer above it
sees nothing but queued dicts, so replacing this in-process thread with a child
process later is a change to this one file.

Pipeline callbacks arrive on pipeline threads and must reach an asyncio queue
owned by the server's loop; that handoff is the entire reason this class exists.
Nothing else may import omnilingual.pipeline.

API keys reach the providers through run_env() -> load_settings(env=...), never
through a flag value: a key in argv is readable by any user on the machine from
`ps`. secrets.py deliberately publishes no reader, so run_env() reads the .env
itself, and the one invariant tests hold it to is that a key the user typed into
that .env is never shadowed by the launch environment — see run_env's docstring
for the precedence decision that guarantees it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from omnilingual.cache import JsonCache
from omnilingual.config import (
    ConfigError,
    Settings,
    load_settings,
    require_keys,
    validate_chunk_bounds,
    validate_target_s,
)
from omnilingual.diarize import build_diarizer
from omnilingual.diarize.base import Diarizer
from omnilingual.pipeline import run_from_chunks, work_dir_for
from omnilingual.pipeline import run as batch_run
from omnilingual.pipeline.live import (
    _HALT_TEXT,
    LiveCapture,
    LiveOptions,
    run_live,
)
from omnilingual.render.markdown import render, render_english_only
from omnilingual.stt import build_stt
from omnilingual.stt.base import STTProvider
from omnilingual.translate import build_translator
from omnilingual.translate.base import Translator
from omnilingual.ui import secrets, settings

log = logging.getLogger(__name__)

# run_live halts by writing one of these marker strings into a stt_failed segment
# and keeps capturing. That marker is the only signal that distinguishes a halted
# run from one that merely had a bad chunk, so it is read from the pipeline's own
# table rather than re-spelled here: a reworded message would silently turn every
# halt back into a plain 'done'. A rename breaks the import loudly instead.
_HALT_MARKERS = frozenset(_HALT_TEXT.values())

# Only run_live was given a stop channel, so only it can be stopped. The batch
# loop and run_from_chunks have no equivalent, and a status that says otherwise
# would be a stop the user is waiting for that never arrives.
_STOPPABLE_MODES = frozenset({"live"})

# The panel's fields, taken from the store that owns them rather than restated
# here: two lists drift, and the drift shows up as a legitimate start request
# rejected for a field the server never mentioned. build_run and live_options
# each read their own subset of this one set, so the same payload goes to both.
_PANEL_FIELDS = frozenset(settings.DEFAULTS)

_ASSIGN = re.compile(
    r"^\s*(?:export\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*?)\s*$"
)


# --- sourcing the environment a run inherits -------------------------------


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _env_file_values() -> dict[str, str]:
    """The .env layer: every assignment it holds, last line per name winning.

    One resolver for the whole UI, so nothing else has to decide what the .env
    means. secrets.py exposes no reader — a key value must never be able to leave
    that module — so the parse is duplicated here, and pinned to
    secrets.present() by tests that assert the same shapes.

    An .env that is not UTF-8 contributes nothing rather than aborting the run,
    which is what secrets.present() does with it too: the file exists, holds keys
    this process never read, and guessing at the readable part is worse than
    treating the layer as empty.
    """
    values: dict[str, str] = {}
    try:
        # newline="" so a CRLF .env is read without its \r being rewritten away,
        # matching secrets._lines().
        with secrets.env_path().open("r", encoding="utf-8", newline="") as fh:
            text = fh.read()
    except (OSError, ValueError):
        return values
    for line in text.split("\n"):
        if match := _ASSIGN.match(line):
            # Last assignment wins and a blank one erases: `tail -n 1`, which is
            # how present() resolves a duplicated key.
            values[match["name"]] = _unquote(match["value"])
    return values


def run_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment a run inherits: the launch environment, with the repo's
    gitignored .env layered over it.

    Precedence is deliberate and is the opposite of the usual convention: **the
    .env wins over a variable already in the process environment.** The .env is
    the file the panel *manages* — secrets.set_key writes it, and the panel edits
    it — so a GUI that let a value exported when the app happened to be launched
    override what the user just typed would silently discard their input. The
    launch environment still fills every name the .env does not mention, which is
    what keeps `uv run --env-file .env` working and what lets a `FOO=bar uv run
    omnilingual-ui` reach a provider that has no key of its own.

    That leaves one intentional asymmetry with secrets.present(), which reports
    the .env layer only: a key that exists *only* in the launch environment is
    usable by a run while the panel calls it unconfigured. That is correct rather
    than a disagreement — the badge answers "is this key in the file I manage",
    and an exported key is a property of how the app was started. The invariant
    that does hold, and that tests pin, is the one that could bite: **a key in
    the .env is never shadowed by the environment.** The panel must never report
    "configured" for a value the run will not use.

    A value never reaches argv, a subprocess, a log line or an event message; it
    goes process memory -> Settings and no further.
    """
    resolved = dict(os.environ if env is None else env)
    resolved.update(_env_file_values())
    return resolved


def build_run(
    values: Mapping[str, object],
    env: Mapping[str, str] | None = None,
) -> tuple[Settings, STTProvider, Translator, Diarizer | None]:
    """Turn panel values into the four objects a run needs, or raise ConfigError.

    Every rule is the CLI's: load_settings rejects an unknown provider, a model
    with no default and a speaker count below two, and the two shared validators
    own the chunk bounds and the live target. None of that is copied here — a
    second copy of a numeric bound is how the panel and the command line start
    disagreeing about what is legal.

    Keys are not parameters. They arrive through run_env() inside load_settings,
    so there is no code path on which a key could be logged, put in a URL, or
    appended to a command line.

    require_keys is called here, before this function returns, and that is the
    whole point of calling it here rather than inside the run. The providers read
    their key lazily, on the first request, so a missing one raises from inside
    pipeline/live.py's worker thread — where live catches QuotaError and AuthError
    and nothing else, so the chunk's result never lands and the coordinator waits
    on it forever. A run with no API key configured then hangs with no end event
    and no way out but force-quitting the app. Raising ConfigError here turns that
    into the 400 it should always have been.

    Every failure this raises is a ConfigError, without exception: an unknown
    field, a numeric field that will not coerce, and every shared validator's own
    complaint alike. The server answers 400, so one except clause has to be enough
    — a second type turns a bad start request into a 500, which is the one thing
    the page cannot explain to the user. The numeric wrapper exists because a JSON
    body is not trustworthy the way a typed CLI argument is: it arrives as text, or
    as null for a field the user cleared, and float(None) is a TypeError the spec
    never mentions.
    """
    unknown = sorted(set(values) - _PANEL_FIELDS)
    if unknown:
        raise ConfigError(f"unknown run field(s): {', '.join(unknown)}")
    mode = str(values.get("mode", "live"))
    min_chunk_s = _numeric(values, "min_chunk_s", 5.0)
    max_chunk_s = _numeric(values, "max_chunk_s", 28.0)
    settings = load_settings(
        env=run_env(env),
        langs=list(values.get("langs") or []),
        min_chunk_s=min_chunk_s,
        max_chunk_s=max_chunk_s,
        # A stored "" means "the provider's own default", which Settings spells
        # None; passing "" through would pin today's spelling into the run.
        stt_provider=str(values.get("stt", "sarvam")),
        stt_model=str(values.get("stt_model") or "") or None,
        mt_provider=str(values.get("mt", "mayura")),
        mt_model=str(values.get("mt_model") or "") or None,
        diarizer="sherpa" if values.get("diarize") else None,
        num_speakers=_numeric(values, "num_speakers", None, as_int=True),
    )
    # Before the providers exist, so an out-of-bounds panel costs nothing and
    # cannot surface as a wall of 400 s from the API. Same order as cli.py:live.
    validate_chunk_bounds(min_chunk_s, max_chunk_s)
    if mode == "live":
        validate_target_s(_numeric(values, "target_s", 8.0),
                          min_chunk_s, max_chunk_s)
    require_keys(settings)
    return (settings, build_stt(settings), build_translator(settings),
            build_diarizer(settings) if settings.diarizer else None)


def _numeric(values: Mapping[str, object], field: str, default,
             *, as_int: bool = False):
    """One panel field as a number, or a ConfigError naming the field.

    A JSON body is not shaped like a typed CLI argument: it carries text where the
    CLI carries numbers, and a field the user cleared arrives as null or as an
    empty string. float()/int() alone would leak ValueError for "abc" and
    TypeError for None, and the spec's 400 path cannot catch either.

    Absent, null and blank all mean "not provided" and fall back to the field's
    default, which is the same truthiness reading config.load_settings applies when
    it turns env.get(...) into a key or None. Anything else is a mistake worth
    naming, so the message says which field and what arrived: the user is looking
    at a form, not at a traceback.
    """
    raw = values.get(field)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        return int(raw) if as_int else float(raw)
    except (TypeError, ValueError) as exc:
        kind = "a whole number" if as_int else "a number"
        raise ConfigError(
            f"--{field.replace('_', '-')} must be {kind} (got {raw!r})"
        ) from exc


def live_options(
    values: Mapping[str, object], *, out: Path, work_root: Path | None = None
) -> LiveOptions:
    """LiveOptions from the panel payload. One place holds the field-to-flag mapping.

    Takes the same payload build_run takes — the whole panel row, unfiltered — so
    the server hands one mapping to both instead of splitting it by hand. It reads
    only the live fields; the rest are build_run's, and neither function rejects
    the other's.

    `out` is a keyword rather than a field because the caller must resolve it
    against a base directory first, and because the collision rename in
    free_output_path has to be applied to the resolved path. A panel `out` is
    therefore ignored here and is not an error.

    work_dir stores "" rather than a literal ".omnilingual" on purpose (see
    ui/settings.py): an empty value has to reach LiveOptions as None so the
    pipeline's own default — .omnilingual beside the output file — applies, and a
    stored ".omnilingual" would instead pin the work tree to the server's
    directory.
    """
    stored = str(values.get("work_dir") or "")
    root = work_root or (Path(stored) if stored else None)
    return LiveOptions(
        out=out,
        device=str(values.get("device") or "Omnilingual"),
        mic_only=bool(values.get("mic_only")),
        target_s=_numeric(values, "target_s", 8.0),
        max_chunk_s=_numeric(values, "max_chunk_s", 28.0),
        min_chunk_s=_numeric(values, "min_chunk_s", 5.0),
        noise_db=_numeric(values, "noise_db", -35.0),
        stt_workers=_numeric(values, "stt_workers", 2, as_int=True),
        max_cost=_numeric(values, "max_cost", 50.0),
        work_root=root,
        english_only=bool(values.get("english_only")),
    )


def free_output_path(path: Path, *, now: datetime | None = None) -> Path:
    """A path no live run is currently writing, renaming only on a collision.

    render/live.py opens the output with O_EXCL, so a second live run aimed at an
    existing path dies with FileExistsError after the start request has already
    answered 200. The CLI sidesteps this for its own --out; the UI has to do the
    same or "transcribe the same file today and again tomorrow" is a hard failure
    the user cannot act on.

    Renamed, not overwritten and not an error: a previous transcript is the user's,
    and clobbering it because the app was started twice would be worse than a
    suffixed filename. The scheme matches cli.py's, which is not shared code —
    that module imports typer and rich, which the UI must not pull in — and this
    is a filename convention rather than a validation rule, so the one thing a
    drift could cost is a different name, never a wrong one.
    """
    if not path.exists():
        return path
    stamp = (now or datetime.now()).strftime("%H%M")
    candidate = path.with_name(f"{path.stem}-{stamp}{path.suffix}")
    serial = 2
    while candidate.exists():
        candidate = path.with_name(
            f"{path.stem}-{stamp}-{serial}{path.suffix}")
        serial += 1
    return candidate


def _is_halt(seg) -> bool:
    """Whether a segment is one of run_live's halt markers.

    Both halves are required. run_live writes the marker text into a stt_failed
    segment and nothing else ever produces that text, so the conjunction cannot
    misfire — but matching the text alone would call a halt any future provider
    error carrying that same string, and matching the status alone would call
    every failed segment a halt, which is the failure this state exists to
    distinguish.
    """
    return seg.status == "stt_failed" and seg.text in _HALT_MARKERS


class SessionRunner:
    """Owns one run's thread, its event stream, and its state.

    Not a singleton and not a registry: one instance per run, several instances
    alive if the caller lets that happen. Only the counters are guarded, and the
    run itself lives in a daemon thread so a quit mid-meeting never waits on
    capture to finish.
    """

    def __init__(self, *, run_id: str, queue,
                 loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.run_id = run_id
        self._queue = queue
        self._loop = loop
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status = "idle"
        self._mode = "live"
        self._started = datetime.now()
        self._out: Path | None = None
        self._session_dir: Path | None = None
        self._segments = 0
        self._dropped = 0
        self._cost = 0.0
        self._cost_cap: float | None = None
        self._seq = 0
        self._exit_code = 0
        self._error: str | None = None
        self._recoverable = False
        self._halted = False

    # --- events ---------------------------------------------------------

    def _emit(self, message: dict) -> None:
        """Hand one event to the server. Callable from any thread.

        A delivery failure is swallowed on purpose. The run belongs to this
        process, not to the page, so a closed loop, a torn-down WebSocket or a
        queue nobody drains must cost the UI its updates and nothing else —
        run_live wraps on_segment for the same reason and status has no such
        guard of its own.
        """
        try:
            if self._loop is None:
                self._queue.put_nowait(message)
            else:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, message)
        except Exception:  # noqa: BLE001 - a dead consumer must not end a run
            log.debug("dropping %s event for run %s", message.get("type"),
                      self.run_id, exc_info=True)

    @staticmethod
    def segment_message(seq: int, seg, cost_inr: float, kept: bool) -> dict:
        """Exactly the spec's §7.1 `segment` shape — one transcript row."""
        return {
            "type": "segment",
            "seq": seq,
            "idx": seg.chunk.idx if seg.chunk.idx >= 0 else seq,
            "start_s": round(seg.chunk.start_s, 2),
            "end_s": round(seg.chunk.end_s, 2),
            "lang": seg.lang,
            "prob": round(seg.prob, 4),
            "text": seg.text,
            "english": seg.english,
            "status": seg.status,
            "speaker": seg.speaker,
            "kept": kept,
            "cost_inr": round(cost_inr, 4),
        }

    def _push_state(self) -> None:
        self._emit({"type": "state", **self.snapshot()})

    # --- lifecycle ------------------------------------------------------

    def start_live(self, *, settings, stt, translator, diarizer,
                   opts: LiveOptions, capture_factory=LiveCapture) -> None:
        self._begin("live")
        self._cost_cap = opts.max_cost
        # Before the thread starts and before the caller reads _out, so the path
        # the start request reports is the path the run actually writes.
        requested = opts.out
        out = free_output_path(requested)
        if out != requested:
            # Name the file that was in the way and the one being written instead.
            # Rebinding opts first would print the new name twice and hide the
            # fact that the filename the user chose was taken.
            opts = replace(opts, out=out)
            self._emit({"type": "log", "level": "warn",
                        "message": f"{requested.name} already exists; "
                                   f"writing {out.name} instead"})
        self._out = out
        self._spawn(self._live, settings, stt, translator, diarizer, opts,
                    capture_factory)

    def start_recording(self, *, settings, stt, translator, diarizer,
                        source: Path, work_root: Path, out: Path,
                        english_only: bool = False) -> None:
        self._begin("recording")
        self._out = out
        self._spawn(self._recording, settings, stt, translator, diarizer, source,
                    work_root, out, english_only)

    def recover(self, *, session_dir: Path, settings, stt, translator,
                diarizer, out: Path, english_only: bool = False) -> None:
        self._begin("recover")
        self._out = out
        self._session_dir = session_dir
        self._spawn(self._recovery, session_dir, settings, stt, translator,
                    diarizer, out, english_only)

    def stop(self) -> None:
        """Ask the run to stop. Idempotent, and safe before a thread exists.

        A no-op for a mode that cannot stop. Only run_live takes a stop channel;
        the batch loop and run_from_chunks have none, so their threads run to
        completion whatever is asked. Flipping their status to "stopping" would
        announce a stop that never happens, leaving the page waiting for an
        interruption that is not coming.
        """
        self._stop.set()
        if self._mode not in _STOPPABLE_MODES:
            return
        with self._lock:
            if self._status == "running":
                self._status = "stopping"
        self._push_state()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # --- starting -------------------------------------------------------

    def _begin(self, mode: str) -> None:
        """Refuse a second run on one instance.

        The registry that keeps a single run at a time belongs to the server, but
        restarting an instance would splice two transcripts into one seq stream,
        so the guard lives where the seq lives.
        """
        if self._thread is not None:
            raise RuntimeError(f"run {self.run_id} has already been started")
        self._mode = mode
        self._started = datetime.now()
        self._status = "running"
        self._halted = False
        # A stop() that arrived before the run did belongs to nothing: carrying it
        # into the run would make the new one stop on its first read, so a runner
        # that was stopped while idle and then started would look broken.
        self._stop.clear()

    def _spawn(self, target, *args) -> None:
        # The state goes out before the thread starts, so the first message the
        # page sees is always the run's own — never a stray status line from a
        # pipeline thread that got there first.
        self._push_state()
        self._thread = threading.Thread(target=target, args=args, daemon=True)
        self._thread.start()

    # --- run bodies -----------------------------------------------------

    def _live(self, settings, stt, translator, diarizer, opts: LiveOptions,
              capture_factory) -> None:
        # run_live reports no session dir, and it mints one itself with a UTC
        # stamp, so the run's directory is found by differencing the work root.
        # Diffing rather than asking keeps this free of shared state, which is
        # what lets two runners exist without either one reaching into the other.
        root = opts.work_root or (opts.out.parent / ".omnilingual")
        before = set(root.glob("live-*")) if root.is_dir() else set()

        def on_status(message: str) -> None:
            self._emit({"type": "status", "message": message})

        def on_segment(seg, keep: bool) -> None:
            if _is_halt(seg):
                self._halted = True
            self._on_segment(seg, keep, 0.0)

        try:
            code = run_live(opts, settings, stt, translator, diarizer=diarizer,
                            status=on_status, capture_factory=capture_factory,
                            on_segment=on_segment, stop_event=self._stop)
        except BaseException as exc:  # noqa: BLE001 - reported, never raised out
            self._discover_session(root, before)
            self._fail(exc)
            return

        self._discover_session(root, before)
        self._exit_code = code
        # Exit code is about the transcript, not about the run: 2 means the file
        # is written and some segments did not come out, which is a valid file.
        # Only 1 is fatal — capture never opened, so there is no file at all.
        if code == 1:
            self._finish("failed")
        else:
            self._finish("halted" if self._halted else "done")

    def _discover_session(self, root: Path, before: set[Path]) -> None:
        after = set(root.glob("live-*")) if root.is_dir() else set()
        # Stamps are %Y%m%dT%H%M%S%fZ, so sorted order is chronological order and
        # the last new directory is this run's.
        fresh = sorted(d for d in after - before if (d / "session.json").is_file())
        if fresh:
            self._session_dir = fresh[-1]

    def _recording(self, settings, stt, translator, diarizer, source: Path,
                   work_root: Path, out: Path, english_only: bool) -> None:
        try:
            work = work_dir_for(source, work_root)
            out.parent.mkdir(parents=True, exist_ok=True)

            def on_progress(index: int, total: int, seg) -> None:
                self._on_segment(seg, True, 0.0)
                self._emit({"type": "log", "level": "info",
                            "message": f"[{index}/{total}] chunks transcribed"})

            transcript = batch_run(source, work, settings, stt, translator,
                                   JsonCache(work / "cache"), on_progress,
                                   diarizer=diarizer)
            self._write(transcript, out, english_only)
            self._cost = transcript.cost.inr_estimate
            self._exit_code = 2 if any(s.status != "ok"
                                       for s in transcript.segments) else 0
        except BaseException as exc:  # noqa: BLE001
            self._fail(exc)
            return
        self._finish("done")

    def _recovery(self, session_dir: Path, settings, stt, translator, diarizer,
                  out: Path, english_only: bool) -> None:
        try:
            out.parent.mkdir(parents=True, exist_ok=True)

            def on_progress(index: int, total: int, seg) -> None:
                self._on_segment(seg, True, 0.0)
                self._emit({"type": "log", "level": "info",
                            "message": f"[{index}/{total}] chunks recovered"})

            transcript = run_from_chunks(session_dir, settings, stt, translator,
                                         JsonCache(session_dir / "cache"),
                                         on_progress, diarizer=diarizer)
            self._write(transcript, out, english_only)
            self._cost = transcript.cost.inr_estimate
            self._exit_code = 2 if any(s.status != "ok"
                                       for s in transcript.segments) else 0
        except BaseException as exc:  # noqa: BLE001
            self._fail(exc)
            return
        self._finish("done")

    def _write(self, transcript, out: Path, english_only: bool) -> None:
        """The file is the artefact, so it is produced by the same renderer the CLI
        uses — the UI renders nothing itself, and a batch transcript is written
        once from the complete Transcript so a failure can never leave a
        half-written file where a valid one or none is the only honest outcome."""
        out.write_text(render(transcript), encoding="utf-8")
        if english_only:
            out.with_suffix(".en.md").write_text(
                render_english_only(transcript), encoding="utf-8")

    # --- shared plumbing -------------------------------------------------

    def _on_segment(self, seg, keep: bool, delta: float) -> None:
        """One sealed chunk, one row.

        `delta` is 0 for both pipeline entry points: run_live's on_segment and the
        batch progress callback carry no cost, and re-deriving one here would put
        a second copy of the pipeline's pricing in the UI. The run's real total
        arrives with the Transcript and lands in the final state.
        """
        with self._lock:
            seq = self._seq
            self._seq += 1
            self._segments += 1
            if keep:
                self._cost += delta
            else:
                self._dropped += 1
        self._emit(self.segment_message(seq, seg, delta, keep))

    def _fail(self, exc: BaseException) -> None:
        log.exception("session %s failed", self.run_id, exc_info=exc)
        # type + message only. A traceback can quote a request header, and this
        # string goes to the page.
        self._error = f"{type(exc).__name__}: {exc}"
        self._exit_code = 1
        self._emit({"type": "log", "level": "error", "message": self._error})
        self._finish("failed")

    def _finish(self, status: str) -> None:
        with self._lock:
            self._status = status
            self._recoverable = self._has_chunks(self._session_dir)
        self._push_state()
        self._emit({
            "type": "end",
            "exit_code": self._exit_code,
            "out": str(self._out) if self._out else None,
            "session_dir": (str(self._session_dir) if self._session_dir
                            else None),
            "recoverable": self._recoverable,
        })

    @staticmethod
    def _has_chunks(session_dir: Path | None) -> bool:
        """Whether this session dir holds anything worth recovering.

        run_live creates the directory and session.json before it opens the
        capture, and chunks.json exists from birth even when empty, so neither
        file proves a sealed chunk. run_from_chunks refuses an empty manifest, and
        a Recover button that always fails is worse than none.
        """
        if session_dir is None:
            return False
        return any((session_dir / "live-chunks").glob("*.wav"))

    # --- read-only views -------------------------------------------------

    @property
    def status(self) -> str:
        return self._status

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    @property
    def out_path(self) -> Path | None:
        return self._out

    @property
    def recoverable(self) -> bool:
        return self._recoverable

    def snapshot(self) -> dict:
        return {
            "run_id": self.run_id,
            "mode": self._mode,
            "status": self._status,
            "elapsed_s": round((datetime.now() - self._started).total_seconds(), 1),
            "cost_inr": round(self._cost, 2),
            # None, not 0.0: only a live run has a cap, and reporting ₹0 would
            # read to the page as a budget that is already spent.
            "cost_cap": self._cost_cap,
            "segments": self._segments,
            "dropped": self._dropped,
            "out": str(self._out) if self._out else None,
            "session_dir": (str(self._session_dir) if self._session_dir
                            else None),
            "error": self._error,
        }


# ConfigError is re-exported so a caller turning a bad start request into a 400
# catches one exception type: build_run, live_options and free_output_path
# raise ConfigError and nothing else, so one except clause is the whole 400 path.
__all__ = [
    "ConfigError",
    "SessionRunner",
    "build_run",
    "free_output_path",
    "live_options",
    "run_env",
]
