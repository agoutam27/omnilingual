"""IndicTrans2 translation to English, running on-device via CTranslate2."""

from __future__ import annotations

import importlib.util
import threading
from collections.abc import Callable
from pathlib import Path

from omnilingual.config import ConfigError, Settings
from omnilingual.translate.mayura import split_text

TARGET_TAG = "eng_Latn"

# IndicTrans2 is prompted with "<src_tag> <tgt_tag> <text>" and asserts both tags
# come from its own script-tagged vocabulary, so Sarvam's xx-IN codes are mapped
# onto it here. kok-IN borrows gom_Deva: Goan Konkani in Devanagari is the only
# Konkani IndicTrans2 was trained on, and ne-IN borrows npi_Deva.
SRC_TAGS: dict[str, str] = {
    "as-IN": "asm_Beng",
    "bn-IN": "ben_Beng",
    "brx-IN": "brx_Deva",
    "doi-IN": "doi_Deva",
    "gu-IN": "guj_Gujr",
    "hi-IN": "hin_Deva",
    "kn-IN": "kan_Knda",
    "kok-IN": "gom_Deva",
    "ks-IN": "kas_Arab",
    "mai-IN": "mai_Deva",
    "ml-IN": "mal_Mlym",
    "mni-IN": "mni_Beng",
    "mr-IN": "mar_Deva",
    "ne-IN": "npi_Deva",
    "od-IN": "ory_Orya",
    "pa-IN": "pan_Guru",
    "sa-IN": "san_Deva",
    "sat-IN": "sat_Olck",
    "sd-IN": "snd_Arab",
    "ta-IN": "tam_Taml",
    "te-IN": "tel_Telu",
    "ur-IN": "urd_Arab",
}

# Beam 4: greedy decoding drifts into repetition loops on the languages IndicTrans2
# handles worst, and beam search was markedly steadier on every one tried.
BEAM_SIZE = 4

TranslateFn = Callable[[list[str], str], list[str]]


class IndicTrans2Translator:
    # Runs on-device, so translation adds nothing to the rupee estimate; the
    # pipeline reads this via getattr like inr_per_hour.
    inr_per_10k_chars = 0.0

    def __init__(self, settings: Settings, *, translate: TranslateFn | None = None) -> None:
        if translate is None:
            for pkg in ("ctranslate2", "sentencepiece", "huggingface_hub"):
                if importlib.util.find_spec(pkg) is None:
                    raise ConfigError(
                        "IndicTrans2 MT needs the local extra: "
                        "uv sync --extra local-mt  # or: uv pip install 'omnilingual[local-mt]'"
                    )
            translate = _CTranslate2Engine(settings.resolved_mt_model)
        self._settings = settings
        self._translate = translate
        self.model = f"indictrans2:{settings.resolved_mt_model}"
        # Live mode's stt_workers threads share this provider; CTranslate2
        # inference is not guaranteed thread-safe for a single model instance.
        self._lock = threading.Lock()

    def supports(self, lang: str) -> bool:
        return lang in SRC_TAGS

    def to_english(self, text: str, src_lang: str) -> str:
        pieces = [p for p in split_text(text, self._settings.mt_char_limit) if p.strip()]
        if not pieces:
            return ""
        with self._lock:
            return " ".join(self._translate(pieces, SRC_TAGS[src_lang]))


class _CTranslate2Engine:
    """Owns the CTranslate2 model and IndicTrans2's SentencePiece tokenizer."""

    def __init__(self, repo_id: str) -> None:
        self._repo_id = repo_id
        self._translator = None
        self._spm = None

    def _ensure_model(self) -> None:
        if self._translator is not None:
            return
        import ctranslate2
        import sentencepiece
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(
            self._repo_id,
            local_dir=Path.home() / ".cache" / "omnilingual" / "models" / "indictrans2",
            # CTranslate2 refuses to open the directory without its config.json and
            # refuses to run without both vocabulary files; model.SRC is the
            # SentencePiece tokenizer that produced the source side of them.
            allow_patterns=["*/ctranslate2_model/*"],
        ))
        # Each direction nests the model under its own folder, so take the one
        # directory instead of hardcoding a path tied to this repo's layout.
        ckpt = next(root.rglob("ctranslate2_model"))
        self._spm = sentencepiece.SentencePieceProcessor(model_file=str(ckpt / "vocab" / "model.SRC"))
        # The published checkpoint is float32; CT2 quantizes as it loads, which is
        # what keeps a 200M-param model small enough to sit alongside a 28 s audio
        # window in 16 GB of RAM.
        self._translator = ctranslate2.Translator(str(ckpt), device="cpu", compute_type="int8")

    def __call__(self, pieces: list[str], src_tag: str) -> list[str]:
        self._ensure_model()
        spm = self._spm
        tokens = [
            [src_tag, TARGET_TAG, *spm.EncodeAsPieces(piece), "</s>"]
            for piece in pieces
        ]
        results = self._translator.translate_batch(tokens, beam_size=BEAM_SIZE)
        return ["".join(r.hypotheses[0]).replace("▁", " ").strip() for r in results]