"""Train and evaluate the fast local baseline on duplicate-safe splits."""

import json
import hashlib
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

from src.baseline_classifier import BaselineGrievanceClassifier
from src.data_splits import has_group_leakage, make_splits, validate_taxonomy_labels


DATA = Path("data/processed/grievances_synthetic.csv")
TAXONOMY = Path("config/taxonomy.json")
OUT = Path("checkpoints/baseline")


def evaluate(clf, frame):
    predictions = [clf.predict(text) for text in frame["text"]]
    true_category = frame["category"].astype(str).tolist()
    true_subcategory = frame["subcategory"].astype(str).tolist()
    true_priority = frame["priority"].astype(int).tolist()
    raw_priority = [float(item["priority_raw"]) for item in predictions]
    decision_priority = [int(item["priority"]) for item in predictions]
    high_true = [int(value >= 4) for value in true_priority]
    high_decision = [int(value >= 4) for value in decision_priority]
    high_raw_cut = [int(value >= 4) for value in raw_priority]
    return {
        "rows": len(frame),
        "category_accuracy": accuracy_score(true_category, [p["category"] for p in predictions]),
        "category_macro_f1": f1_score(true_category, [p["category"] for p in predictions], average="macro"),
        "subcategory_macro_f1": f1_score(true_subcategory, [p["subcategory"] for p in predictions], average="macro"),
        "priority_mae_raw": mean_absolute_error(true_priority, raw_priority),
        "priority_recall_high": recall_score(high_true, high_decision, zero_division=0),
        "priority_precision_high": precision_score(high_true, high_decision, zero_division=0),
        "priority_recall_high_raw_cut4": recall_score(high_true, high_raw_cut, zero_division=0),
        "priority_true_distribution": {str(k): int(v) for k, v in sorted(Counter(true_priority).items())},
        "priority_prediction_distribution": {
            str(k): int(v) for k, v in sorted(Counter(decision_priority).items())
        },
        "high_priority_definition": "priority >= 4",
    }


def main():
    df = pd.read_csv(DATA)
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    validate_taxonomy_labels(df, taxonomy)
    train_df, validation_df, test_df = make_splits(df, seed=42)
    if has_group_leakage((train_df, validation_df, test_df)):
        raise RuntimeError("Duplicate or exact-text leakage detected between data splits")

    clf = BaselineGrievanceClassifier(TAXONOMY)
    clf.fit(train_df["text"], train_df["category"], train_df["subcategory"], train_df["priority"])
    threshold_info = clf.calibrate_threshold(
        validation_df["text"], validation_df["category"], min_coverage=0.60
    )
    clf.model_version = "tfidf_baseline_synthetic_v2"

    # Freeze the operating threshold on validation before evaluating the test split.
    validation_metrics = evaluate(clf, validation_df)
    test_metrics = evaluate(clf, test_df)
    metrics = {
        "dataset_kind": "synthetic_demo_pipeline_only",
        "research_claim": "not_vcet_pilot_results",
        "split_seed": 42,
        "split_sizes": {"train": len(train_df), "validation": len(validation_df), "test": len(test_df)},
        "duplicate_group_leakage": False,
        "confidence_threshold": threshold_info,
        "validation": validation_metrics,
        "test": test_metrics,
        "routing_accuracy": None,
        "routing_accuracy_note": "Unavailable: no validated gold department column in this dataset.",
    }
    clf.save(OUT)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    training_config = {
        "model_version": clf.model_version,
        "model_identifier": "tfidf_logistic_regression_plus_ridge_priority",
        "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "taxonomy_sha256": hashlib.sha256(TAXONOMY.read_bytes()).hexdigest(),
        "scikit_learn_version": version("scikit-learn"),
        "split_seed": 42,
        "split_sizes": metrics["split_sizes"],
        "duplicate_group_leakage": False,
    }
    (OUT / "training_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
