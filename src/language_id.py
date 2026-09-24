"""Language/script detection and privacy-first PII redaction.

The project scope is English, Devanagari Hindi and Romanised Hinglish.
A fastText model can be supplied for language identification, but Romanised
Hinglish is additionally checked with a small, auditable marker lexicon because
standard language-ID models often collapse Latin-script Hindi into English.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from src.configuration import load_threshold_settings

ROLL_NUMBER_PATTERN = re.compile(r"\b[A-Z]{2,5}[- ]?\d{4,10}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)")
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

# Conservative markers: words that are disproportionately common in
# Romanised Hindi. They are used only as a fallback/augmentation signal.
HINGLISH_MARKERS = {
    "abhi", "acha", "accha", "aap", "apna", "apni", "bahut", "bohot", "bhi",
    "chahiye", "chal", "dene", "dena", "diya", "doosra", "hai", "hain", "ho",
    "hoga", "ka", "ke", "ki", "kya", "kyu", "kyunki", "lag", "mujhe", "mera",
    "meri", "mere", "nahi", "nahi", "nahin", "par", "pe", "raha", "rahi", "sakta",
    "se", "tha", "theek", "toh", "wala", "wali", "yaha", "ye", "yeh", "yaar",
    "kripya", "please", "kar", "karo", "karna", "nahi", "mila", "mil", "problem",
}


def _romanised_hinglish_score(text: str) -> float:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in HINGLISH_MARKERS)
    return hits / max(len(words), 1)


@lru_cache(maxsize=1)
def _load_fasttext(model_path: str | None):
    if not model_path:
        return None
    try:
        import fasttext
    except Exception:
        return None
    if not Path(model_path).exists():
        return None
    try:
        return fasttext.load_model(model_path)
    except Exception:
        return None


def detect_language(text: str) -> str:
    """Return ``en``, ``hi`` or ``hinglish``.

    Devanagari is treated as Hindi for the project's supported labels. For
    Latin-script text, an optional fastText model is consulted and then a
    Hinglish marker score is used so code-mixed Romanised Hindi is not treated
    as plain English by default.
    """
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"

    fasttext_model = _load_fasttext(os.getenv("FASTTEXT_MODEL_PATH"))
    if fasttext_model is not None:
        try:
            labels, probs = fasttext_model.predict(text.replace("\n", " "), k=2)
            top_label = labels[0].replace("__label__", "")
            top_prob = float(probs[0])
            if top_label in {"hi", "hin"} and top_prob >= load_threshold_settings()["fasttext_hindi_threshold"]:
                return "hinglish"
        except Exception:
            pass

    return "hinglish" if _romanised_hinglish_score(text) >= 0.08 else "en"


def detect_language_details(text: str) -> dict[str, object]:
    """Return a best-effort language/script label and heuristic confidence.

    The confidence is a rule-strength indicator, not a calibrated probability.
    It is exposed for transparency and must be calibrated before research use.
    """
    if not text or not text.strip():
        return {"language": "en", "script": "unknown", "confidence": 0.0}
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if devanagari:
        script = "mixed" if latin else "devanagari"
        ratio = devanagari / max(devanagari + latin, 1)
        return {"language": "hi", "script": script, "confidence": round(min(.99, .65 + .34 * ratio), 3)}
    score = _romanised_hinglish_score(text)
    lang = detect_language(text)
    return {
        "language": lang,
        "script": "latin",
        "confidence": round(min(.95, .55 + min(score, .4)) if lang == "hinglish" else .7, 3),
    }


def _load_name_list() -> list[str]:
    path = os.getenv("STUDENT_NAMES_FILE", "")
    if not path or not Path(path).exists():
        return []
    try:
        names = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name:
                names.append(name)
        return sorted(names, key=len, reverse=True)
    except Exception:
        return []


@lru_cache(maxsize=1)
def _compiled_name_patterns():
    return [re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE) for name in _load_name_list()]


def redact_pii(text: str) -> str:
    """Redact direct identifiers before DB storage or external API calls.

    Name redaction is enabled only when an institution supplies an explicit name-list
    file; the system does not attempt to guess names from arbitrary text.
    """
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = PHONE_PATTERN.sub("[PHONE]", text)
    text = ROLL_NUMBER_PATTERN.sub("[ROLL_NUMBER]", text)
    for pattern in _compiled_name_patterns():
        text = pattern.sub("[NAME]", text)
    return text
