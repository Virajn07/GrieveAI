"""Load and validate non-secret demo settings from repository configuration."""

import json
import os
from pathlib import Path


def load_threshold_settings() -> dict[str, float]:
    path = Path(os.getenv("THRESHOLDS_CONFIG", "config/thresholds.json"))
    config = json.loads(path.read_text(encoding="utf-8"))
    confidence = float(config["confidence_threshold"])
    duplicate = float(config["duplicate_similarity_threshold"])
    fasttext_hindi = float(config["fasttext_hindi_threshold"])
    if not 0 <= confidence <= 1 or not 0 <= duplicate <= 1 or not 0 <= fasttext_hindi <= 1:
        raise ValueError("configured thresholds must be between 0 and 1")
    return {
        "confidence_threshold": confidence,
        "duplicate_similarity_threshold": duplicate,
        "fasttext_hindi_threshold": fasttext_hindi,
    }
