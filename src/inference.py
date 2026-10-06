"""Production-facing, UI-independent grievance inference API.

The Flask layer can call :func:`predict_grievance` without importing training
code. A lightweight baseline remains the default so local use does not require
GPU or transformer downloads; a saved MuRIL adapter can be selected through
``MODEL_BACKEND=muril``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.baseline_classifier import BaselineGrievanceClassifier
from src.pipeline import analyze_grievance

logger = logging.getLogger(__name__)
_classifier: Any | None = None
_classifier_key: tuple[str, str, str, str] | None = None


def get_classifier() -> Any:
    """Load and cache the configured inference model.

    A missing optional transformer checkpoint is logged and falls back to the
    checked-in baseline, matching the app's local-first behavior.
    """
    global _classifier, _classifier_key
    backend = os.getenv("MODEL_BACKEND", "baseline").strip().lower()
    taxonomy_path = os.getenv("TAXONOMY_CONFIG", "config/taxonomy.json")
    baseline_path = os.getenv("BASELINE_CHECKPOINT_DIR", "checkpoints/baseline")
    transformer_path = os.getenv("MURIL_CHECKPOINT_DIR", "checkpoints/muril_lora/run1")
    key = (backend, taxonomy_path, baseline_path, transformer_path)
    if _classifier is not None and _classifier_key == key:
        return _classifier

    if backend == "muril":
        try:
            from src.model import MuRILInference

            _classifier = MuRILInference(transformer_path, taxonomy_path)
            _classifier_key = key
            return _classifier
        except Exception:
            logger.exception("MuRIL model unavailable; using the local baseline")
    elif backend != "baseline":
        raise ValueError(f"Unsupported MODEL_BACKEND: {backend!r}")

    _classifier = BaselineGrievanceClassifier.load(baseline_path, taxonomy_path)
    _classifier_key = key
    return _classifier


def predict_grievance(
    text: str,
    *,
    classifier: Any | None = None,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """Redact and classify one grievance, returning a serializable result.

    The original input is never included in the result. Explanations and LLM
    calls are intentionally opt-in and are handled by their own services.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    model = classifier or get_classifier()
    analysis = analyze_grievance(text, model, confidence_threshold)
    prediction = analysis["prediction"]
    return {
        "redacted_text": analysis["text"],
        "language": analysis["language"]["language"],
        "script": analysis["language"]["script"],
        "language_confidence": analysis["language"]["confidence"],
        "category": prediction["category"],
        "subcategory": prediction["subcategory"],
        "priority": int(prediction["priority"]),
        "priority_raw": prediction.get("priority_raw"),
        "confidence": float(prediction["confidence"]),
        "subcategory_confidence": prediction.get("subcategory_confidence"),
        "confidence_threshold": float(analysis["threshold"]),
        "requires_human_review": bool(analysis["manual_review"]),
        "model_version": getattr(model, "model_version", model.__class__.__name__),
    }


def reset_classifier_cache() -> None:
    """Clear the process cache; useful when configuration changes in tests."""
    global _classifier, _classifier_key
    _classifier = None
    _classifier_key = None
