"""Render a Transcript to Markdown. Pure functions, no I/O."""

from __future__ import annotations

from collections import defaultdict

from omnilingual.models import Segment, Transcript

_NOTES = {
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
        by_lang[seg.lang] += seg.chunk.duration_s
    total = sum(by_lang.values()) or 1.0
    ranked = sorted(by_lang.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(lang, round(100 * dur / total)) for lang, dur in ranked]


def _header(t: Transcript, suffix: str = "") -> list[str]:
    return [f"# Meeting transcript — {t.source.name}{suffix}", ""]


def render(t: Transcript) -> str:
    lines = _header(t)
    shares = ", ".join(f"{lang} {pct}%" for lang, pct in lang_share(t.segments))
    lines.append(
        f"Duration {fmt_ts(t.duration_s)} · {len(t.segments)} segments · Languages: {shares}"
    )
    lines.append(f"Estimated cost: ₹{t.cost.inr_estimate:.2f}")
    lines += ["", "## Transcript", ""]
    for seg in t.segments:
        head = f"**[{fmt_ts(seg.chunk.start_s)} → {fmt_ts(seg.chunk.end_s)}] {seg.lang}**"
        if seg.status != "ok":
            head += f" _({_NOTES[seg.status]})_"
        lines.append(head)
        lines.append(seg.text)
        if seg.status == "ok" and seg.english and seg.lang != "en-IN":
            lines.append(f"> {seg.english}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def render_english_only(t: Transcript) -> str:
    lines = _header(t, " (English)")
    for seg in t.segments:
        if seg.status == "stt_failed":
            lines.append("[transcription failed]")
        elif seg.lang == "en-IN":
            lines.append(seg.text)
        elif seg.english:
            lines.append(seg.english)
        else:
            lines.append(f"[{seg.lang}, untranslated]")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
