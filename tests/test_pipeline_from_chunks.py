import json
from pathlib import Path

import pytest

from omnilingual.cache import JsonCache
from omnilingual.config import load_settings
from omnilingual.models import STTResult, chunks_to_json, Chunk
from omnilingual.pipeline import LIVE_SESSION_KIND, run_from_chunks

from tests.conftest import make_wav


class FakeSTT:
    model = "saaras:v4"
    mode = "transcribe"

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, wav_path: Path) -> STTResult:
        self.calls += 1
        return STTResult(lang="hi-IN", prob=0.97, text="नमस्ते सब लोग")


class FakeTranslator:
    model = "mayura:v1"

    def supports(self, lang: str) -> bool:
        return lang != "en-IN" and lang != "unknown"

    def to_english(self, text: str, src_lang: str) -> str:
        return "Hello everyone"


def make_session(tmp_path: Path) -> Path:
    sd = tmp_path / "live-20260905T153000Z"
    chunks_dir = sd / "live-chunks"
    chunks_dir.mkdir(parents=True)
    chunks = []
    # Distinct durations so the two wavs differ: identical bytes would share one
    # content-addressed cache key and correctly cost a single STT call.
    for i, dur in enumerate((6.0, 5.0)):
        wav = chunks_dir / f"{i:04d}.wav"
        make_wav(wav, [("tone", dur)])
        chunks.append(Chunk(idx=i, start_s=i * 7.0, end_s=i * 7.0 + dur, wav_path=wav))
    (sd / "session.json").write_text(
        json.dumps({"kind": LIVE_SESSION_KIND, "version": 1, "device": "Omnilingual", "mic_only": False}),
        encoding="utf-8",
    )
    (sd / "chunks.json").write_text(chunks_to_json(chunks), encoding="utf-8")
    return sd


def test_run_from_chunks_transcribes_manifest(tmp_path):
    sd = make_session(tmp_path)
    settings = load_settings(api_key="k", env={})
    stt, translator = FakeSTT(), FakeTranslator()
    t = run_from_chunks(sd, settings, stt, translator, JsonCache(sd / "cache"))

    assert stt.calls == 2
    assert [s.status for s in t.segments] == ["ok", "ok"]
    assert all(s.english == "Hello everyone" for s in t.segments)
    assert t.duration_s == t.segments[-1].chunk.end_s
    assert t.source == Path(sd.name)


def test_run_from_chunks_second_run_pays_nothing(tmp_path):
    sd = make_session(tmp_path)
    settings = load_settings(api_key="k", env={})
    cache = JsonCache(sd / "cache")
    run_from_chunks(sd, settings, FakeSTT(), FakeTranslator(), cache)

    stt2 = FakeSTT()
    t2 = run_from_chunks(sd, settings, stt2, FakeTranslator(), cache)
    assert stt2.calls == 0  # everything answered by the session cache
    assert [s.status for s in t2.segments] == ["ok", "ok"]


def test_run_from_chunks_rejects_non_session_dir(tmp_path):
    settings = load_settings(api_key="k", env={})
    with pytest.raises(ValueError, match="not a live session dir"):
        run_from_chunks(tmp_path, settings, FakeSTT(), FakeTranslator(), JsonCache(tmp_path / "cache"))


def test_run_from_chunks_rejects_wrong_kind(tmp_path):
    sd = tmp_path / "live-x"
    sd.mkdir()
    (sd / "session.json").write_text(json.dumps({"kind": "something-else"}), encoding="utf-8")
    settings = load_settings(api_key="k", env={})
    with pytest.raises(ValueError, match="not a live session dir"):
        run_from_chunks(sd, settings, FakeSTT(), FakeTranslator(), JsonCache(sd / "cache"))


def test_run_from_chunks_rejects_missing_wav(tmp_path):
    sd = make_session(tmp_path)
    (sd / "live-chunks" / "0001.wav").unlink()
    settings = load_settings(api_key="k", env={})
    with pytest.raises(ValueError, match="missing 1 chunk wav"):
        run_from_chunks(sd, settings, FakeSTT(), FakeTranslator(), JsonCache(sd / "cache"))


def test_run_from_chunks_rejects_empty_manifest(tmp_path):
    sd = make_session(tmp_path)
    (sd / "chunks.json").write_text("[]", encoding="utf-8")
    settings = load_settings(api_key="k", env={})
    with pytest.raises(ValueError, match="no sealed chunks"):
        run_from_chunks(sd, settings, FakeSTT(), FakeTranslator(), JsonCache(sd / "cache"))
