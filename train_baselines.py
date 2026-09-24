"""Compare the keyword-rule and TF-IDF + SVM baselines on a leakage-aware split."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from src.baselines import KeywordRuleClassifier, TfidfSVMClassifier

DATA = Path("data/processed/grievances_synthetic.csv")
TAXONOMY = Path("config/taxonomy.json")


def group_key(row):
    if str(row["duplicate_of"]).strip() and str(row["duplicate_of"]) != "nan":
        return str(row["duplicate_of"])
    normalized = " ".join(str(row["text"]).lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


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


def main():
    df = pd.read_csv(DATA)
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    df["group"] = df.apply(group_key, axis=1)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(splitter.split(df, df["category"], df["group"]))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    keyword = KeywordRuleClassifier(taxonomy)
    kw_preds = [(lambda x: (*x, 3.0))(keyword.predict(t)) for t in val_df["text"]]

    svm = TfidfSVMClassifier(taxonomy).fit(
        train_df["text"], train_df["category"], train_df["subcategory"], train_df["priority"]
    )
    svm_preds = [svm.predict(t) for t in val_df["text"]]

    results = {
        "dataset_kind": "synthetic_demo_pipeline_validation",
        "research_claim": "not_vcet_pilot_results",
        "split": {"train_rows": len(train_df), "validation_rows": len(val_df), "duplicate_group_leakage": False},
        "keyword_rules": evaluate(val_df, kw_preds),
        "tfidf_svm": evaluate(val_df, svm_preds),
        "routing_accuracy": None,
        "routing_accuracy_note": "Unavailable: no validated gold department column in this dataset.",
        "override_rate": None,
        "override_rate_note": "Unavailable in offline model evaluation: no observed human routing decisions.",
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
