"""Map Whisper's language identifiers to the BCP-47 codes the pipeline speaks.

Whisper reports ISO 639-1 codes ("hi"), the OpenAI-style verbose_json that Groq
serves reports full lowercase names ("hindi"), and omnilingual's pipeline keys on
Sarvam-style codes ("hi-IN" — note Odia is "od-IN", not "or-IN"). Anything
unmapped passes through unchanged so a surprise code stays visible in output.
"""

from __future__ import annotations

_WHISPER_TO_BCP47: dict[str, str] = {
    "as": "as-IN", "assamese": "as-IN",
    "bn": "bn-IN", "bengali": "bn-IN",
    "brx": "brx-IN", "bodo": "brx-IN",
    "doi": "doi-IN", "dogri": "doi-IN",
    "gu": "gu-IN", "gujarati": "gu-IN",
    "hi": "hi-IN", "hindi": "hi-IN",
    "kn": "kn-IN", "kannada": "kn-IN",
    "ks": "ks-IN", "kashmiri": "ks-IN",
    "kok": "kok-IN", "konkani": "kok-IN",
    "mai": "mai-IN", "maithili": "mai-IN",
    "ml": "ml-IN", "malayalam": "ml-IN",
    "mni": "mni-IN", "manipuri": "mni-IN",
    "mr": "mr-IN", "marathi": "mr-IN",
    "ne": "ne-IN", "nepali": "ne-IN",
    "or": "od-IN", "odia": "od-IN",
    "pa": "pa-IN", "punjabi": "pa-IN",
    "sa": "sa-IN", "sanskrit": "sa-IN",
    "sat": "sat-IN", "santali": "sat-IN",
    "sd": "sd-IN", "sindhi": "sd-IN",
    "ta": "ta-IN", "tamil": "ta-IN",
    "te": "te-IN", "telugu": "te-IN",
    "ur": "ur-IN", "urdu": "ur-IN",
    "en": "en-IN", "english": "en-IN",
}


def to_bcp47(code: str) -> str:
    if not code or not code.strip():
        return "unknown"
    code = code.strip().lower()
    return _WHISPER_TO_BCP47.get(code, code)
