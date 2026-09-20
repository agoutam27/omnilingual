"""Local Whisper speech-to-text on Apple Silicon via the optional mlx-whisper extra."""

from __future__ import annotations

import importlib.util
import threading
import wave
from collections.abc import Callable
from pathlib import Path

from omnilingual.config import ConfigError, Settings
from omnilingual.models import STTResult
from omnilingual.stt.langs import to_bcp47

DecodeFn = Callable[[Path], "tuple[str, str, float]"]


class MlxWhisperSTT:
    """Whisper running fully on-device: zero marginal cost, works offline.

    decode is injectable so unit tests never touch the model weights.
    """

    mode = "transcribe"
    inr_per_hour = 0.0

    def __init__(self, settings: Settings, *, decode: DecodeFn | None = None) -> None:
        if decode is None:
            if importlib.util.find_spec("mlx_whisper") is None:
                raise ConfigError(
                    "mlx-whisper STT needs the local extra: "
                    "uv pip install 'omnilingual[local-stt]'"
                )
            decode = _MlxEngine(settings.resolved_stt_model)
        self._decode = decode
        self.model = f"mlx-whisper:{settings.resolved_stt_model}"
        # Live mode's stt_workers threads share this provider; MLX inference is
        # not thread-safe, so all calls serialize here.
        self._lock = threading.Lock()

    def transcribe(self, wav_path: Path) -> STTResult:
        with self._lock:
            text, lang, prob = self._decode(wav_path)
        return STTResult(lang=to_bcp47(lang), prob=prob, text=text.strip())


class _MlxEngine:
    """Owns the Whisper model and runs an explicit language-detection pass per
    chunk: the high-level transcribe() returns the detected language but drops
    its probability, which STTResult needs."""

    def __init__(self, repo_id: str) -> None:
        self._repo_id = repo_id
        self._model = None
        self._mx = None

    def _ensure_model(self):
        if self._model is None:
            import mlx.core as mx
            from mlx_whisper.transcribe import ModelHolder

            self._mx = mx
            self._model = ModelHolder.get_model(self._repo_id, mx.float16)
        return self._model

    def __call__(self, wav_path: Path) -> tuple[str, str, float]:
        import mlx_whisper
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

        model = self._ensure_model()
        audio = _load_wav(wav_path)
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        mel_segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(self._mx.float16)
        _, probs = model.detect_language(mel_segment)
        lang = max(probs, key=probs.get)
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=self._repo_id, language=lang, verbose=None
        )
        return result["text"], lang, float(probs[lang])


def _load_wav(wav_path: Path):
    """Decode the pipeline's canonical 16 kHz mono int16 WAV with the stdlib;
    anything unexpected falls back to mlx-whisper's ffmpeg-based loader."""
    try:
        with wave.open(str(wav_path), "rb") as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, 16000):
                raise ValueError("not canonical 16 kHz mono int16")
            frames = w.readframes(w.getnframes())
    except (wave.Error, ValueError, EOFError):
        from mlx_whisper.audio import load_audio

        return load_audio(str(wav_path))
    import numpy as np

    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
