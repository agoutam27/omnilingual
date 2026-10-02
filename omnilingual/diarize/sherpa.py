"""Local speaker diarization via sherpa-onnx (pure ONNX runtime, Intel-Mac friendly)."""

from __future__ import annotations

import importlib.util
import sys
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from omnilingual.config import ConfigError, Settings
from omnilingual.diarize.base import Turn

SEGMENTATION = "pyannote-segmentation-3-0"
EMBEDDING = "3dspeaker-eres2net"

SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)

Downloader = Callable[[str, Path], None]


def _default_model_dir() -> Path:
    return Path.home() / ".cache" / "omnilingual" / "models"


def _mb(n: int) -> str:
    return f"{n / 1_000_000:.1f} MB"


class _DownloadProgress:
    """Throttled stderr progress for a model fetch.

    `live --diarize` builds the tracker before it prints anything, so a silent
    ~46 MB download reads as a hang. Updates are throttled to ~1/sec: a
    carriage-return line on a TTY, discrete lines when stderr is redirected.
    """

    _interval = 1.0

    def __init__(self, label: str, total: int | None) -> None:
        self._label = label
        self._total = total
        self._done = 0
        self._started = time.monotonic()
        self._last = 0.0
        self._tty = sys.stderr.isatty()
        self._prev = ""

    def start(self) -> None:
        size = f" ({_mb(self._total)})" if self._total else ""
        sys.stderr.write(f"downloading {self._label}{size}\n")
        sys.stderr.flush()

    def advance(self, n: int) -> None:
        self._done += n
        now = time.monotonic()
        complete = self._total is not None and self._done >= self._total
        if not complete and now - self._last < self._interval:
            return
        self._last = now
        line = self._render(now)
        if self._tty:
            sys.stderr.write("\r" + line + " " * max(0, len(self._prev) - len(line)))
            self._prev = line
        else:
            sys.stderr.write(line + "\n")
        sys.stderr.flush()

    def close(self) -> None:
        if self._tty and self._prev:
            sys.stderr.write("\n")
            self._prev = ""
        sys.stderr.flush()

    def _render(self, now: float) -> str:
        rate = self._done / max(now - self._started, 1e-6) / 1_000_000
        if not self._total:
            return f"  {self._label} {_mb(self._done)} at {rate:.2f} MB/s"
        pct = 100.0 * self._done / self._total
        return f"  {self._label} {pct:.0f}% {_mb(self._done)}/{_mb(self._total)} at {rate:.2f} MB/s"


def _http_download(url: str, dest: Path) -> None:
    """Stream a URL to dest atomically; extracts the segmentation tarball."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(dest.parent), suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as resp:
            resp.raise_for_status()
            length = resp.headers.get("content-length")
            progress = _DownloadProgress(url.rsplit("/", 1)[-1], int(length) if length else None)
            progress.start()
            try:
                with tmp_path.open("wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
                        progress.advance(len(chunk))
            finally:
                progress.close()
        if url.endswith(".tar.bz2"):
            with tempfile.TemporaryDirectory(dir=str(dest.parent)) as td:
                with tarfile.open(tmp_path, "r:bz2") as tar:
                    tar.extractall(td, filter="data")
                extracted = next(Path(td).rglob("model.onnx"))
                extracted.replace(dest)
        else:
            tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)


class SherpaDiarizer:
    """Offline diarization running fully on-device: zero marginal cost, works offline.

    engine/downloader are injectable so unit tests never touch models or network.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        engine: Callable[[Path], list[Turn]] | None = None,
        downloader: Downloader | None = None,
        model_dir: Path | None = None,
    ) -> None:
        if engine is None:
            if importlib.util.find_spec("sherpa_onnx") is None:
                raise ConfigError(
                    "sherpa-onnx diarization needs the local extra: "
                    "uv sync --extra diarize"
                )
            engine = _SherpaEngine(
                settings,
                model_dir=model_dir or _default_model_dir(),
                downloader=downloader or _http_download,
            )
        self._engine = engine
        self.model = f"sherpa:{SEGMENTATION}-{EMBEDDING}"
        # Live mode's stt_workers threads share this diarizer; the ONNX session
        # is not guaranteed thread-safe, so all calls serialize here.
        self._lock = threading.Lock()

    def ensure_models(self) -> list[Path]:
        if isinstance(self._engine, _SherpaEngine):
            return self._engine.ensure_models()
        return []

    def chunk_embedder(self) -> Callable[[Path], object]:
        """Per-chunk embedding callable for live tracking (real engine only)."""
        if not isinstance(self._engine, _SherpaEngine):
            raise ConfigError(
                "live speaker tracking needs the sherpa models, not a test engine"
            )
        return self._engine.embedder()

    def diarize(self, wav_path: Path) -> list[Turn]:
        with self._lock:
            return list(self._engine(wav_path))


class _SherpaEngine:
    """Owns the sherpa-onnx diarizer: downloads models once, reuses the session."""

    def __init__(
        self, settings: Settings, *, model_dir: Path, downloader: Downloader
    ) -> None:
        self._num_speakers = settings.num_speakers
        self._model_dir = model_dir
        self._downloader = downloader
        self._diarizer = None

    @property
    def _seg_path(self) -> Path:
        return self._model_dir / SEGMENTATION / "model.onnx"

    @property
    def _emb_path(self) -> Path:
        return self._model_dir / (EMBEDDING + ".onnx")

    def ensure_models(self) -> list[Path]:
        paths = [self._seg_path, self._emb_path]
        urls = [SEGMENTATION_URL, EMBEDDING_URL]
        for path, url in zip(paths, urls):
            if not path.is_file():
                self._downloader(url, path)
        return paths

    def _ensure_diarizer(self):
        if self._diarizer is None:
            import sherpa_onnx

            seg_path, emb_path = self.ensure_models()
            if self._num_speakers:
                clustering = sherpa_onnx.FastClusteringConfig(
                    num_clusters=self._num_speakers
                )
            else:
                clustering = sherpa_onnx.FastClusteringConfig(threshold=0.5)
            config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=str(seg_path)
                    ),
                ),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(emb_path)
                ),
                clustering=clustering,
                min_duration_on=0.3,
                min_duration_off=0.5,
            )
            if not config.validate():
                raise ConfigError(
                    "sherpa-onnx diarization config invalid; "
                    "re-run with a fresh model download"
                )
            self._diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
        return self._diarizer

    def __call__(self, wav_path: Path) -> list[Turn]:
        import soundfile as sf

        diarizer = self._ensure_diarizer()
        samples, _rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
        # process() returns an OfflineSpeakerDiarizationResult, which is not
        # iterable; sort_by_start_time() is what hands back the segment list.
        segments = diarizer.process(samples).sort_by_start_time()
        return [
            Turn(start_s=float(s.start), end_s=float(s.end), speaker=str(s.speaker))
            for s in segments
        ]

    def embedder(self) -> Callable[[Path], object]:
        """Lazily built per-chunk embedding callable for live tracking."""
        _seg_path, emb_path = self.ensure_models()
        extractor = None

        def embed(wav_path: Path) -> object:
            nonlocal extractor
            if extractor is None:
                import sherpa_onnx

                extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
                    sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                        model=str(emb_path)
                    )
                )
            import soundfile as sf

            samples, rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
            # compute() takes an OnlineStream, not samples; feeding one WAV is
            # create_stream -> accept_waveform -> input_finished -> compute.
            stream = extractor.create_stream()
            stream.accept_waveform(rate, samples)
            stream.input_finished()
            return extractor.compute(stream)

        return embed
