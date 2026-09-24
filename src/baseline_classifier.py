"""Strong local baseline: TF-IDF + Logistic Regression + Ridge priority.

This model keeps the application runnable without a GPU while matching the
same interface used by the MuRIL+LoRA backend. It is also the classical
baseline for the evaluation chapter.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import joblib
import numpy as np
from src.configuration import load_threshold_settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge


class BaselineGrievanceClassifier:
    def __init__(self, taxonomy_path="config/taxonomy.json"):
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            self.taxonomy = json.load(f)
        self.vectorizer = TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            strip_accents=None,
        )
        self.category_clf = LogisticRegression(max_iter=1500, class_weight="balanced")
        self.subcategory_clf = LogisticRegression(max_iter=1500, class_weight="balanced")
        self.priority_reg = Ridge(alpha=1.0)
        self.is_fitted = False
        self.default_confidence_threshold = load_threshold_settings()["confidence_threshold"]
        self.confidence_threshold = self.default_confidence_threshold

    def fit(self, texts: Iterable[str], categories, subcategories, priorities):
        X = self.vectorizer.fit_transform(texts)
        self.category_clf.fit(X, list(categories))
        self.subcategory_clf.fit(X, list(subcategories))
        self.priority_reg.fit(X, np.asarray(list(priorities), dtype=float))
        self.is_fitted = True
        return self

    def predict(self, text: str) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Classifier is not fitted. Train or load checkpoints/baseline first.")
        X = self.vectorizer.transform([text])

        category_probs = self.category_clf.predict_proba(X)[0]
        cat_idx = int(np.argmax(category_probs))
        category = str(self.category_clf.classes_[cat_idx])
        confidence = float(category_probs[cat_idx])

        # Hierarchical inference: restrict the sub-category choice to labels
        # valid for the predicted category instead of using a flat argmax.
        valid_subs = self.taxonomy["categories"][category]["subcategories"]
        sub_probs = self.subcategory_clf.predict_proba(X)[0]
        sub_classes = list(self.subcategory_clf.classes_)
        candidates = [
            (str(label), float(prob))
            for label, prob in zip(sub_classes, sub_probs)
            if label in valid_subs
        ]
        subcategory, sub_confidence = max(candidates, key=lambda x: x[1])

        priority_raw = float(self.priority_reg.predict(X)[0])
        priority = int(np.clip(np.rint(priority_raw), 1, 5))

        return {
            "category": category,
            "subcategory": subcategory,
            "priority": priority,
            "priority_raw": round(priority_raw, 3),
            "confidence": round(confidence, 3),
            "subcategory_confidence": round(sub_confidence, 3),
        }

    def explain(self, text: str, top_k: int = 8):
        """Exact linear-model token/ngram contributions for the prediction."""
        X = self.vectorizer.transform([text])
        category = self.category_clf.predict(X)[0]
        class_idx = list(self.category_clf.classes_).index(category)
        coefs = self.category_clf.coef_[class_idx]
        feature_names = np.asarray(self.vectorizer.get_feature_names_out())
        nonzero_idx = X.nonzero()[1]
        contributions = [
            (str(feature_names[i]), float(coefs[i] * X[0, i])) for i in nonzero_idx
        ]
        contributions.sort(key=lambda x: -x[1])
        return contributions[:top_k]

    def calibrate_threshold(self, texts, labels, min_coverage=0.60):
        """Choose the lowest threshold that reaches the requested validation
        precision while retaining at least ``min_coverage`` of examples.
        Falls back to the configured threshold when a useful operating point cannot be found.
        """
        X = self.vectorizer.transform(list(texts))
        probs = np.max(self.category_clf.predict_proba(X), axis=1)
        preds = self.category_clf.classes_[np.argmax(self.category_clf.predict_proba(X), axis=1)]
        labels = np.asarray(list(labels))
        best = None
        for threshold in np.arange(0.35, 0.91, 0.01):
            routed = probs >= threshold
            coverage = float(np.mean(routed))
            if coverage < min_coverage:
                continue
            precision = float(np.mean(preds[routed] == labels[routed])) if routed.any() else 0.0
            candidate = (precision, coverage, float(threshold))
            if best is None or candidate > best:
                best = candidate
        self.confidence_threshold = round(best[2], 2) if best else self.default_confidence_threshold
        return {
            "threshold": self.confidence_threshold,
            "routed_coverage": round(best[1], 3) if best else None,
            "routed_precision": round(best[0], 3) if best else None,
        }

    def save(self, path="checkpoints/baseline"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(path, "vectorizer.joblib"))
        joblib.dump(self.category_clf, os.path.join(path, "category_clf.joblib"))
        joblib.dump(self.subcategory_clf, os.path.join(path, "subcategory_clf.joblib"))
        joblib.dump(self.priority_reg, os.path.join(path, "priority_reg.joblib"))
        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"confidence_threshold": self.confidence_threshold}, f, indent=2)

    @classmethod
    def load(cls, path="checkpoints/baseline", taxonomy_path="config/taxonomy.json"):
        obj = cls(taxonomy_path)
        obj.vectorizer = joblib.load(os.path.join(path, "vectorizer.joblib"))
        obj.category_clf = joblib.load(os.path.join(path, "category_clf.joblib"))
        obj.subcategory_clf = joblib.load(os.path.join(path, "subcategory_clf.joblib"))
        priority_path = os.path.join(path, "priority_reg.joblib")
        if os.path.exists(priority_path):
            obj.priority_reg = joblib.load(priority_path)
        else:
            # Backward compatibility with the original repo checkpoint.
            obj.priority_reg = None
        meta_path = os.path.join(path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                obj.confidence_threshold = float(json.load(f).get("confidence_threshold", obj.confidence_threshold))
        obj.is_fitted = True
        return obj
