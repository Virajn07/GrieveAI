"""UI-independent preprocessing, inference, and confidence-gate service."""

from __future__ import annotations

from typing import Any

from src.configuration import load_threshold_settings
from src.language_id import detect_language_details, redact_pii


def analyze_grievance(raw_text: str, classifier: Any, confidence_threshold: float | None = None) -> dict[str, Any]:
    """Redact identifiers, classify, and decide whether routing needs review.

    The returned text is always the redacted version. A configured threshold
    takes precedence; otherwise a validation-calibrated model threshold is
    used. This service makes no database or Flask calls.
    """
    text = redact_pii(raw_text)
    language = detect_language_details(text)
    prediction = classifier.predict(text)
    threshold = confidence_threshold
    if threshold is None:
        threshold = getattr(classifier, "confidence_threshold", None)
    if threshold is None:
        threshold = load_threshold_settings()["confidence_threshold"]
    threshold = float(threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("confidence threshold must be between 0 and 1")
    return {
        "text": text,
        "language": language,
        "prediction": prediction,
        "threshold": threshold,
        "manual_review": float(prediction["confidence"]) < threshold,
    }
