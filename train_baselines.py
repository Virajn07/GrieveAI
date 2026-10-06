"""Compare keyword and TF-IDF + SVM baselines on the model's fixed split."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, recall_score

from src.baselines import KeywordRuleClassifier, TfidfSVMClassifier
from src.data_splits import has_group_leakage, make_splits, validate_taxonomy_labels

DATA = Path("data/processed/grievances_synthetic.csv")
TAXONOMY = Path("config/taxonomy.json")


def evaluate(true_df, preds):
    return {
        "category_macro_f1": f1_score(true_df["category"], [p[0] for p in preds], average="macro"),
        "subcategory_macro_f1": f1_score(true_df["subcategory"], [p[1] for p in preds], average="macro"),
        "priority_mae": mean_absolute_error(true_df["priority"].astype(float), [p[2] for p in preds]),
        "priority_recall_high": recall_score(
            (true_df["priority"].astype(float) >= 4).astype(int),
            [int(p[2] >= 4) for p in preds],
            zero_division=0,
        ),
    }


def evaluate_pair(frame, keyword, svm):
    keyword_predictions = [(*keyword.predict(text), 3.0) for text in frame["text"]]
    svm_predictions = [svm.predict(text) for text in frame["text"]]
    return {
        "keyword_rules": evaluate(frame, keyword_predictions),
        "tfidf_svm": evaluate(frame, svm_predictions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = pd.read_csv(DATA)
    taxonomy_json = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    validate_taxonomy_labels(df, taxonomy_json)
    taxonomy = {"categories": taxonomy_json["categories"]}
    train_df, val_df, test_df = make_splits(df, seed=args.seed)

    keyword = KeywordRuleClassifier(taxonomy)
    svm = TfidfSVMClassifier(taxonomy).fit(
        train_df["text"], train_df["category"], train_df["subcategory"], train_df["priority"]
    )
    results = {
        "dataset_kind": "synthetic_demo_pipeline_validation",
        "research_claim": "not_vcet_pilot_results",
        "split": {
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "duplicate_group_leakage": has_group_leakage((train_df, val_df, test_df)),
        },
        "validation": evaluate_pair(val_df, keyword, svm),
        "test": evaluate_pair(test_df, keyword, svm),
        "routing_accuracy": None,
        "routing_accuracy_note": "Unavailable: no validated gold department column in this dataset.",
        "override_rate": None,
        "override_rate_note": "Unavailable in offline model evaluation: no observed human routing decisions.",
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
