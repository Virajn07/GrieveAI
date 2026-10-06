"""Production-facing, UI-independent grievance inference API.

IndicBERT v2 + LoRA is the primary configured model. TF-IDF is available only
when explicitly selected with ``MODEL_BACKEND=baseline``. Missing transformer
assets raise a clear error instead of silently changing the model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.baseline_classifier import BaselineGrievanceClassifier
from src.pipeline import analyze_grievance

_classifier: Any | None = None
_classifier_key: tuple[str, str, str, str, str] | None = None


def get_classifier() -> Any:
    """Load and cache the configured inference model.

    The primary IndicBERT backend and explicit alternatives are never swapped
    silently. Set ``MODEL_BACKEND=baseline`` to deliberately use TF-IDF.
    """
    global _classifier, _classifier_key
    backend = os.getenv("MODEL_BACKEND", "indicbert").strip().lower()
    taxonomy_path = os.getenv("TAXONOMY_CONFIG", "config/taxonomy.json")
    baseline_path = os.getenv("BASELINE_CHECKPOINT_DIR", "checkpoints/baseline")
    indicbert_path = os.getenv("INDICBERT_CHECKPOINT_DIR", "checkpoints/indicbert_lora/run1")
    muril_path = os.getenv("MURIL_CHECKPOINT_DIR", "checkpoints/muril_lora/run1")
    key = (backend, taxonomy_path, baseline_path, indicbert_path, muril_path)
    if _classifier is not None and _classifier_key == key:
        return _classifier

    if backend == "indicbert":
        from src.indicbert_inference import IndicBERTInference

        _classifier = IndicBERTInference(indicbert_path, taxonomy_path)
        _classifier_key = key
        return _classifier

    if backend == "muril":
        from src.model import MuRILInference

        _classifier = MuRILInference(muril_path, taxonomy_path)
        _classifier_key = key
        return _classifier
    if backend != "baseline":
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
