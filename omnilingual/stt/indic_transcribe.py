"""Local Indic-Transcribe-core speech-to-text via the optional indic-stt extra."""

from __future__ import annotations

import importlib.util
import sys
import threading
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

from omnilingual.config import ConfigError, Settings
from omnilingual.models import STTResult
from omnilingual.stt.langs import from_bcp47, to_bcp47

DecodeFn = Callable[[Path], "tuple[str, str, float]"]

# Import names of what the model's own code needs (see its requirements.txt).
_REQUIRED = ("torch", "torchaudio", "transformers", "sentencepiece", "soundfile", "huggingface_hub")


class IndicTranscribeSTT:
    """Bodhan AI's 1.2B Canary-derived model, on-device: free, offline, 25 languages
    incl. English, language ID built in (restricted to --langs when given).

    decode is injectable so unit tests never touch the model weights.
    """

    # "native" script output; the model's "mixed" mode transcribed identically on the
    # code-mixed audio we measured, so there is no reason to pay for a second knob.
    mode = "native"
    inr_per_hour = 0.0

    def __init__(self, settings: Settings, *, decode: DecodeFn | None = None) -> None:
        if decode is None:
            missing = [m for m in _REQUIRED if importlib.util.find_spec(m) is None]
            if missing:
                raise ConfigError(
                    f"indic-transcribe STT needs the indic-stt extra (missing: {', '.join(missing)}): "
                    "uv sync --extra indic-stt  # or: uv pip install 'omnilingual[indic-stt]'"
                )
            decode = _IndicEngine(settings.resolved_stt_model, settings.langs)
        self._decode = decode
        self.model = f"indic-transcribe:{settings.resolved_stt_model}"
        # Live mode's stt_workers threads share this provider; one model instance on
        # one device cannot run concurrent generate() calls safely.
        self._lock = threading.Lock()

    def transcribe(self, wav_path: Path) -> STTResult:
        with self._lock:
            text, lang, prob = self._decode(wav_path)
        return STTResult(lang=to_bcp47(lang), prob=prob, text=text.strip())


class _IndicEngine:
    """Loads the model eagerly: the repo is gated, and a missing licence or token is
    better reported when the backend is built than on the first chunk mid-run."""

    def __init__(self, model_id: str, langs: Sequence[str]) -> None:
        import torch

        model_dir = self._resolve(model_id)
        # The wrapper ships inside the model repo, not on PyPI (see its README).
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from indic_transcribe import LANGUAGES, IndicTranscribe

        # One warning per chunk whose detected language is hi/ur/...: noise in a
        # pipeline that already restricts the candidate set via --langs.
        warnings.filterwarnings("ignore", message=r"identified '.*', one of this model's weakest")
        # MPS (Metal) runs ~36x realtime vs ~4x on CPU; fp32 because bf16 is unproven there.
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._asr = IndicTranscribe.from_pretrained(model_dir, device=device, dtype=dtype)
        # A hard filter: audio in an excluded language is reassigned to the nearest
        # permitted one, which is what --langs asks for. Codes the model was not
        # trained on are dropped rather than failing the run.
        allowed = [c for c in (from_bcp47(code) for code in langs) if c in LANGUAGES]
        self._allowed = allowed or None

    @staticmethod
    def _resolve(model_id: str) -> str:
        if Path(model_id).is_dir():
            return model_id
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

        try:
            return snapshot_download(model_id)
        except (GatedRepoError, RepositoryNotFoundError) as e:
            raise ConfigError(
                f"cannot download {model_id}: the repo is gated. Accept its licence on "
                f"huggingface.co/{model_id} and run `hf auth login` ({type(e).__name__})."
            ) from e

    def __call__(self, wav_path: Path) -> tuple[str, str, float]:
        text, lid = self._asr.transcribe(
            str(wav_path), lang=None, mode="native", return_lid=True, allowed_langs=self._allowed
        )
        return text, lid["lang"], float(lid["topk"][0][1])
