"""Load and validate non-secret demo settings from repository configuration."""

import json
import os
from pathlib import Path


def load_threshold_settings() -> dict[str, float]:
    path = Path(os.getenv("THRESHOLDS_CONFIG", "config/thresholds.json"))
    config = json.loads(path.read_text(encoding="utf-8"))
    confidence_override = os.getenv("ML_CONFIDENCE_THRESHOLD")
    if confidence_override is None:
        confidence_override = os.getenv("CONFIDENCE_THRESHOLD")
    confidence = float(confidence_override) if confidence_override not in (None, "") else float(config["confidence_threshold"])
    duplicate_override = os.getenv("DEDUPE_THRESHOLD")
    duplicate = float(duplicate_override) if duplicate_override not in (None, "") else float(config["duplicate_similarity_threshold"])
    related = float(os.getenv("RELATED_SIMILARITY_THRESHOLD", config.get("related_similarity_threshold", 0.65)))
    recurring = float(os.getenv("RECURRING_SIMILARITY_THRESHOLD", config.get("recurring_similarity_threshold", 0.80)))
    fasttext_hindi = float(config["fasttext_hindi_threshold"])
    if not 0 <= confidence <= 1 or not 0 <= related <= duplicate <= 1 or not 0 <= recurring <= 1 or not 0 <= fasttext_hindi <= 1:
        raise ValueError("configured thresholds must be between 0 and 1")
    return {
        "confidence_threshold": confidence,
        "duplicate_similarity_threshold": duplicate,
        "related_similarity_threshold": related,
        "recurring_similarity_threshold": recurring,
        "fasttext_hindi_threshold": fasttext_hindi,
    }
