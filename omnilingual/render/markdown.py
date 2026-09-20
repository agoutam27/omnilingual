"""Render a Transcript to Markdown. Pure functions, no I/O."""

from __future__ import annotations

from collections import defaultdict

from omnilingual.models import Segment, Transcript

_NOTES = {
    "no_speech": "no speech detected",
    "stt_failed": "transcription failed",
    "mt_unsupported": "translation unavailable: language not supported by translator",
    "mt_failed": "translation failed",
}


def fmt_ts(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def lang_share(segments: list[Segment]) -> list[tuple[str, int]]:
    if not segments:
        return []
    by_lang: dict[str, float] = defaultdict(float)
    for seg in segments:
        if seg.status == "no_speech":
            # The detected language of silence says nothing about who spoke.
            continue
        by_lang[seg.lang] += seg.chunk.duration_s
    total = sum(by_lang.values()) or 1.0
    ranked = sorted(by_lang.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(lang, round(100 * dur / total)) for lang, dur in ranked]


def speaker_share(segments: list[Segment]) -> list[tuple[str, int]]:
    if not segments:
        return []
    by_speaker: dict[str, float] = defaultdict(float)
    for seg in segments:
        if seg.status == "no_speech":
            # Silence contributes speaking time to nobody, even if a stale label lingers.
            continue
        if not seg.speaker:
            continue
        by_speaker[seg.speaker] += seg.chunk.duration_s
    total = sum(by_speaker.values()) or 1.0
    ranked = sorted(by_speaker.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(speaker, round(100 * dur / total)) for speaker, dur in ranked]


def _header(t: Transcript, suffix: str = "") -> list[str]:
    return [f"# Meeting transcript — {t.source.name}{suffix}", ""]


def format_segment(seg: Segment) -> list[str]:
    """Markdown lines for one segment, shared verbatim by the batch renderer and
    the live incremental writer. Always ends with the blank separator line."""
    speaker = f"{seg.speaker} · " if seg.speaker else ""
    head = f"**[{fmt_ts(seg.chunk.start_s)} → {fmt_ts(seg.chunk.end_s)}] {speaker}{seg.lang}**"
    if seg.status != "ok":
        head += f" _({_NOTES[seg.status]})_"
    lines = [head]
    if seg.status != "no_speech":
        lines.append(seg.text)
    if seg.status == "ok" and seg.english and seg.lang != "en-IN":
        lines.append(f"> {seg.english}")
    lines.append("")
    return lines


def format_english_line(seg: Segment) -> str | None:
    """The English-only rendering of one segment, or None when it contributes
    nothing (silence). Mirrors the batch render_english_only rules exactly."""
    if seg.status == "no_speech":
        return None
    if seg.status == "stt_failed":
        line = "[transcription failed]"
    elif seg.lang == "en-IN":
        line = seg.text
    elif seg.english:
        line = seg.english
    else:
        line = f"[{seg.lang}, untranslated]"
    return f"{seg.speaker}: {line}" if seg.speaker else line


def render(t: Transcript) -> str:
    lines = _header(t)
    shares = ", ".join(f"{lang} {pct}%" for lang, pct in lang_share(t.segments))
    lines.append(
        f"Duration {fmt_ts(t.duration_s)} · {len(t.segments)} segments · Languages: {shares}"
    )
    lines.append(f"Estimated cost: ₹{t.cost.inr_estimate:.2f}")
    speaker_shares = ", ".join(f"{speaker} {pct}%" for speaker, pct in speaker_share(t.segments))
    if speaker_shares:
        lines.append(f"Speakers: {speaker_shares}")
    lines += ["", "## Transcript", ""]
    for seg in t.segments:
        lines += format_segment(seg)
    return "\n".join(lines).rstrip("\n") + "\n"


def render_english_only(t: Transcript) -> str:
    lines = _header(t, " (English)")
    for seg in t.segments:
        line = format_english_line(seg)
        if line is None:
            continue
        lines.append(line)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
