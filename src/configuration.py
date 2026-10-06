"""Load and validate non-secret demo settings from repository configuration."""

import json
import os
from pathlib import Path


def load_threshold_settings() -> dict[str, float]:
    path = Path(os.getenv("THRESHOLDS_CONFIG", "config/thresholds.json"))
    config = json.loads(path.read_text(encoding="utf-8"))

    def configured_float(name: str, default: float) -> float:
        value = os.getenv(name, "").strip()
        return float(value) if value else float(default)

    confidence = configured_float(
        "ML_CONFIDENCE_THRESHOLD",
        configured_float("CONFIDENCE_THRESHOLD", config["confidence_threshold"]),
    )
    duplicate = configured_float("DEDUPE_THRESHOLD", config["duplicate_similarity_threshold"])
    related = configured_float("RELATED_SIMILARITY_THRESHOLD", config.get("related_similarity_threshold", 0.65))
    recurring = configured_float("RECURRING_SIMILARITY_THRESHOLD", config.get("recurring_similarity_threshold", 0.80))
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
