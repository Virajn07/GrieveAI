"""Train the fast local baseline used by the Flask MVP."""

import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, recall_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from src.baseline_classifier import BaselineGrievanceClassifier


DATA = Path("data/processed/grievances_synthetic.csv")
TAXONOMY = Path("config/taxonomy.json")
OUT = Path("checkpoints/baseline")


def main():
    df = pd.read_csv(DATA)
    df["group"] = df["duplicate_of"].fillna(df["id"]).astype(str)
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(split.split(df, df["category"], df["group"]))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    clf = BaselineGrievanceClassifier(TAXONOMY)
    clf.fit(train_df["text"], train_df["category"], train_df["subcategory"], train_df["priority"])
    threshold_info = clf.calibrate_threshold(val_df["text"], val_df["category"], min_coverage=0.60)

    preds = [clf.predict(t) for t in val_df["text"]]
    cat_true = val_df["category"].tolist()
    cat_pred = [p["category"] for p in preds]
    sub_true = val_df["subcategory"].tolist()
    sub_pred = [p["subcategory"] for p in preds]
    pri_true = val_df["priority"].astype(float).tolist()
    pri_pred = [p["priority_raw"] for p in preds]
    high_true = [int(x >= 4) for x in pri_true]
    high_pred = [int(x >= 4) for x in pri_pred]

    metrics = {
        "dataset_kind": "synthetic_demo_pipeline_validation",
        "research_claim": "not_vcet_pilot_results",
        "category_accuracy": accuracy_score(cat_true, cat_pred),
        "category_macro_f1": f1_score(cat_true, cat_pred, average="macro"),
        "subcategory_macro_f1": f1_score(sub_true, sub_pred, average="macro"),
        "priority_mae": mean_absolute_error(pri_true, pri_pred),
        "priority_recall_high": recall_score(high_true, high_pred, zero_division=0),
        "routing_accuracy": None,
        "routing_accuracy_note": "Unavailable: no validated gold department column in this dataset.",
        "override_rate": None,
        "override_rate_note": "Unavailable in offline model evaluation: no observed human routing decisions.",
        "confidence_threshold": threshold_info,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
    }
    clf.save(OUT)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
