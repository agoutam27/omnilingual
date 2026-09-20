# Speaker Diarization — Research & Integration Design

Date: 2026-09-20
Status: Tasks 1–6 implemented on `feat/speaker-diarization` (plan `2026-09-20-speaker-diarization.md`); Phase 4 eval gate still open
Audience: omnilingual maintainers
Builds on: `2026-09-19-stt-provider-alternatives-design.md`

## 1. Context & Goal

The transcript currently identifies languages but not people. Goal: label each
segment with a stable, anonymous speaker tag — `Speaker 1`, `Speaker 2`, … — in
both the batch `transcribe` output and the growing `live` transcript.

Numbering is **first-appearance order** within one recording (Speaker 1 = first
voice heard). No cross-recording voiceprints, no real names — that is a
commercial feature (pyannoteAI precision-2) and explicitly out of scope.

## 2. Integration Points (verified 2026-09-20)

| Piece | Location | Consequence |
|---|---|---|
| `Segment` dataclass | `omnilingual/models.py` | gains `speaker: str | None = None` — additive, existing constructors keep working |
| `format_segment()` | `omnilingual/render/markdown.py:42` | **single shared renderer** for batch and live; head becomes `**[00:01:23 → 00:01:31] Speaker 2 · hi-IN**`; one edit covers both outputs |
| `format_english_line()` | `omnilingual/render/markdown.py:57` | English-only file lines become `Speaker 2: <text>` |
| `transcribe_chunks()` | `omnilingual/pipeline/__init__.py` | batch assignment point — speaker attached when the Segment is built |
| `process_chunk()` | `omnilingual/pipeline/live.py` | live assignment point, runs on worker threads → live diarizer must be thread-safe |
| `prepare()` / `work_dir_for()` | `omnilingual/pipeline/__init__.py` | work dir is already keyed by the recording's sha256 — diarization turns cache as `diarize-<model>.json` beside `chunks.json`; rerun = free |
| `conftest.py` | `tests/` | add `diarize` mark to the opt-in pattern (`--run-diarize`), auto-skipped by default like `live`/`mlx`/`faster_whisper` |

## 3. Research: Diarization Options (2026)

Diarization models are **language-agnostic** (speaker embeddings, not text), so
Indian-language meetings need no Indic-specific model — but the embedding models
below are trained mostly on zh/en speech, so §6.4 validates on real meetings.

| Option | License | Deps | DER (benchmark) | Speed | Fit |
|---|---|---|---|---|---|
| **sherpa-onnx** (k2-fsa): pyannote-segmentation-3-0 ONNX + 3D-Speaker ERes2Net embedding + FastClustering | Apache-2.0 | `sherpa-onnx` pip, **onnxruntime only** — already shipped for faster-whisper; models ≈40 MB from GitHub releases, no token | no published DER on this exact combo; segmentation is pyannote 3.0-class | fast on CPU (seg+embed ≪ real-time); est. minutes per hour of audio | **Primary** — runs on the Intel i9 today, mac x86_64 wheels exist |
| **pyannote community-1** (Sep 2025, pyannote.audio 4.x) | CC-BY-4.0, **HF gated token** | torch + pyannote.audio (~2 GB) | best open-source: AMI-IHM 17.0, AliMeeting 20.3, DIHARD3 20.2, VoxConverse 11.2 | 31 s/hr on H100; **≈1× RT on Intel CPU → a 1 h meeting ≈ 1 h** | Optional quality extra, off by default |
| pyannote 3.1 (legacy) | MIT, gated | torch | strictly worse than community-1 everywhere | same | skip — superseded |
| NVIDIA Sortformer v2 / streaming | CC-BY-4.0 | NeMo (heavy) | competitive, **hard limit 4 speakers**, long-form needs chunk stitching | 214× RT | skip — meetings exceed 4 speakers |
| DiariZen (BUT-FIT) | research OSS | torch, WavLM | 13.3 avg, strong 5+ speakers | — | watch-list, immature packaging |
| pyannoteAI hosted (precision-2 / community-1 API) | commercial | httpx only | best overall (11.2); voiceprints | cloud | possible later behind same protocol |

**Decision: sherpa-onnx primary, pyannote community-1 as an optional extra,
both behind one `Diarizer` protocol** — mirroring the STT-provider pattern.

## 4. Design

### 4.1 Protocol & data model

```python
# omnilingual/diarize/base.py
@dataclass(frozen=True)
class Turn:
    start_s: float
    end_s: float
    speaker: int          # cluster id, renumbered to first-appearance order later

class Diarizer(Protocol):
    model: str
    def diarize(self, wav_path: Path) -> list[Turn]: ...
```

- `Segment.speaker: str | None` added in `models.py`.
- `assign_speakers(turns, chunks) -> dict[int, str]`: per chunk, the turn with
  **maximum temporal overlap** wins; a chunk with no overlapping turn gets
  `None` (rendered without a label). Cluster ids are renumbered to
  first-appearance order so labels read Speaker 1, 2, 3 in chronological order.
- Mixed chunks (speaker change inside one chunk) attribute all text to the
  dominant speaker — acceptable at 5–28 s granularity; word-timestamp
  refinement is future work (§7).

### 4.2 Batch flow (transcribe)

```
prepare() → normalized.wav
    └─ DiarizeOnce: diarize(normalized.wav) → turns
       cached at <work_dir>/diarize-<model>.json (json list of turns)
       → assign per chunk → transcribe_chunks(..., speakers=map)
```

- Diarization runs **once per recording**, before or in parallel with the STT
  loop; the cache makes reruns and `--estimate` free.
- Turn JSON cache follows the existing `chunking.json` validity pattern
  (recompute when the diarizer `model` string changes).

### 4.3 Live flow

Full-recording diarization can't run mid-meeting, so live uses the **same
models, different strategy** — incremental centroid clustering:

1. Per sealed chunk (already 16 kHz mono WAV), extract one speaker embedding
   with sherpa-onnx's `SpeakerEmbeddingExtractor` (the same ERes2Net model).
2. Maintain per-speaker centroid embeddings; cosine-similarity assign if
   `sim ≥ threshold`, else mint a new speaker. First-appearance numbering
   falls out naturally.
3. One `threading.Lock` — the STT provider pattern already established; live
   `--stt-workers` stays 1 when diarizing.
4. **Drift correction at recovery**: `--from-chunks` re-runs full offline
   diarization over the session audio and relabels everything — the saved
   session becomes the ground truth. Live labels are best-effort, recovery
   labels are authoritative.

Threshold and centroid update rule (EMA vs. running mean) are constants tuned
in §6.4, not CLI flags.

### 4.4 CLI & config

- `transcribe --diarize` (flag, default off), `--speakers N` (optional hint;
  omit → FastClustering threshold mode auto-counts), `--diarizer {sherpa,pyannote}` (default sherpa).
- `live --diarize` same semantics; documents `--stt-workers 1` recommendation.
- `Settings` gains `diarizer: str = "none"`, `num_speakers: int | None = None`.
- pyproject extra: `diarize = ["sherpa-onnx>=1.12", "soundfile"]`; pyannote as
  `diarize-pyannote = ["pyannote.audio>=4"]` + `HF_TOKEN` env. Core deps stay
  httpx/typer/rich.
- sherpa-onnx models (segmentation ~6 MB + ERes2Net ~34 MB) download on first
  use from k2-fsa GitHub releases into `~/.cache/omnilingual/models/` —
  prefetch helper in `scripts/setup-mac.sh`.

### 4.5 Rendering

```
**[00:01:23 → 00:01:31] Speaker 2 · hi-IN**
<text>
> <english>
```

- No label when `speaker is None` (silence, undiarized) — identical to today.
- Header gains a speaker share line mirroring `lang_share`:
  `Speakers: Speaker 1 41%, Speaker 2 33%, Speaker 3 26%` (speaking-time
  share over non-silence segments); live header recomputes on each append.
- English-only file: `Speaker 2: <text>` per line.

## 5. Testing

- Unit: `assign_speakers` edge cases — empty turns, chunk straddling two turns
  (max-overlap wins, tie → earlier turn), gaps between turns → `None`,
  renumbering order; render tests with/without speaker; live centroid
  clustering with synthetic embeddings (two well-separated clusters, one
  ambiguous → new speaker); cache validity on model change.
- Contract: fake `Diarizer` injected into `transcribe_chunks` — segments carry
  speakers without touching STT.
- Opt-in real-model test (`pytest.mark.diarize`, auto-skipped by conftest):
  synthetic two-speaker wav (alternate two TTS/sine-voice clips) → sherpa
  returns ≥2 clusters and both chunk labels.
- Existing suite must stay green — `speaker=None` path is today's behavior.

## 6. Phased Plan

### Phase 1 — Batch diarization (sherpa-onnx)
1. `models.Segment.speaker`, render changes + tests.
2. `diarize/base.py` (Turn, Diarizer), `diarize/sherpa.py` (model download,
   `OfflineSpeakerDiarization`, thread-safe lazy init, find_spec guard with
   install hint — same pattern as `MlxWhisperSTT`).
3. `assign_speakers` + pipeline wiring + turns cache + `--diarize/--speakers`.
4. Tests per §5. **Mergeable alone.**

### Phase 2 — Live incremental labels
1. `diarize/online.py`: embedding extractor + centroid clustering + lock.
2. Wire into `process_chunk`; `--from-chunks` recovery re-diarizes offline and
   relabels (authoritative pass).
3. Tests with synthetic embeddings + a scripted live session.

### Phase 3 — Optional pyannote community-1 provider
1. `diarize/pyannote_impl.py` behind `diarize-pyannote` extra + `HF_TOKEN`.
2. Document CPU cost (~1× RT) — batch overnight use, never live on Intel.

### Phase 4 — Evaluation
1. Extend `scripts/eval_stt.py` (or sibling `eval_diarize.py`): run a real
   meeting with/without diarization; human spot-check 20 random segments for
   correct speaker; check speaker-count plausibility vs known attendees.
2. Tune cluster threshold on 2–3 archived meetings; record constants in doc.
3. If sherpa quality fails on Indian-accented speech (embedding models are
   zh/en-trained), fall back to Phase 3 pyannote as the default and note it.

## 7. Risks & Future Work

- **R-1 Embedding quality on Indian-accented speech** — unverified; Phase 4 gate.
- **R-2 Mixed-speaker chunks** get one dominant label; future: faster-whisper
  word timestamps + turn boundaries to split chunk text by speaker.
- **R-3 Live label drift** (speaker 2 early becomes speaker 3 late) — mitigated
  by recovery relabeling; residual risk accepted for v1.
- **R-4 Overlapped speech** (two people at once) — v1 assigns dominant speaker;
  sherpa exposes overlap via segmentation scores if we later want a
  "(crosstalk)" marker.
- **R-5 sherpa-onnx wheel for macOS x86_64** — listed as supported; Phase 1
  confirms on this machine before building further.
- Future: cross-meeting voiceprints (who is this?), pyannoteAI hosted
  precision-2 behind the same `Diarizer` protocol.

## 8. Sources

- sherpa-onnx diarization docs & Python example: k2-fsa.github.io/sherpa/onnx/speaker-diarization, github.com/k2-fsa/sherpa-onnx (Apache-2.0; macOS x64 ✔)
- sherpa-onnx model zoo: github.com/k2-fsa/sherpa-onnx/releases (speaker-segmentation-models, speaker-recongition-models)
- pyannote community-1: huggingface.co/pyannote/speaker-diarization-community-1 (CC-BY-4.0, DER table, exclusive-diarization mode), pyannote.ai/blog/community-1
- Diarization benchmark survey: arXiv:2509.26177 (Sep 2025 — PyannoteAI 11.2, DiariZen 13.3, Sortformer v2 RTF 214×, 4-speaker limit)
- Repo internals: `omnilingual/render/markdown.py`, `omnilingual/render/live.py`, `omnilingual/pipeline/`, `omnilingual/stt/` (read 2026-09-20)
