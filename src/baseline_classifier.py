"""
Lightweight baseline classifier: TF-IDF + Logistic Regression.

Why this exists: it gives the Flask app a REAL, working end-to-end
prediction pipeline today - no GPU, no model download, trains in
seconds. This is NOT the model that goes in the evaluation section of
the report; src/model.py (MuRIL + LoRA) is trained separately in Colab
and is the target model. Once that checkpoint exists, app/routes.py
swaps BaselineGrievanceClassifier for it - the predict()/explain()
interface below is designed to match, so that's a small change, not a
rewrite.

Bonus: a TF-IDF/LogReg vs MuRIL+LoRA comparison is a legitimate,
easy-to-defend ablation for the evaluation chapter ("does the
transformer actually buy accuracy over a classical baseline, and by
how much") - keep the numbers this script prints, you'll want them later.
"""

import json
import os

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class BaselineGrievanceClassifier:
    def __init__(self, taxonomy_path="config/taxonomy.json"):
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            self.taxonomy = json.load(f)
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.category_clf = LogisticRegression(max_iter=1000)
        self.subcategory_clf = LogisticRegression(max_iter=1000)
        self.priority_clf = LogisticRegression(max_iter=1000)  # priority 1-5 treated as classes
        self.is_fitted = False

    def fit(self, texts, categories, subcategories, priorities):
        X = self.vectorizer.fit_transform(texts)
        self.category_clf.fit(X, categories)
        self.subcategory_clf.fit(X, subcategories)
        self.priority_clf.fit(X, [str(p) for p in priorities])
        self.is_fitted = True

    def predict(self, text: str) -> dict:
        X = self.vectorizer.transform([text])

        category = self.category_clf.predict(X)[0]
        confidence = float(np.max(self.category_clf.predict_proba(X)[0]))

        subcategory = self.subcategory_clf.predict(X)[0]
        # Enforce the taxonomy hierarchy: if the flat sub-category model
        # picked something that doesn't belong under the predicted
        # category, fall back to the best-scoring VALID sub-category.
        valid_subs = self.taxonomy["categories"][category]["subcategories"]
        if subcategory not in valid_subs:
            sub_proba = self.subcategory_clf.predict_proba(X)[0]
            sub_classes = self.subcategory_clf.classes_
            valid_scores = [(c, p) for c, p in zip(sub_classes, sub_proba) if c in valid_subs]
            subcategory = max(valid_scores, key=lambda x: x[1])[0] if valid_scores else valid_subs[0]

        priority = int(self.priority_clf.predict(X)[0])

        return {
            "category": category,
            "subcategory": subcategory,
            "priority": priority,
            "confidence": round(confidence, 3),
        }

    def explain(self, text: str, top_k: int = 5):
        """Top_k words in THIS text that pushed the model toward its
        predicted category. Exact (not approximate, unlike SHAP on a
        deep model) because Logistic Regression is linear - a genuine
        stand-in explainability method until the MuRIL+SHAP path is built."""
        X = self.vectorizer.transform([text])
        category = self.category_clf.predict(X)[0]
        class_idx = list(self.category_clf.classes_).index(category)
        coefs = self.category_clf.coef_[class_idx]
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        nonzero_idx = X.nonzero()[1]
        contributions = [(feature_names[i], float(coefs[i] * X[0, i])) for i in nonzero_idx]
        contributions.sort(key=lambda x: -x[1])
        return contributions[:top_k]

    def save(self, path="checkpoints/baseline"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(path, "vectorizer.joblib"))
        joblib.dump(self.category_clf, os.path.join(path, "category_clf.joblib"))
        joblib.dump(self.subcategory_clf, os.path.join(path, "subcategory_clf.joblib"))
        joblib.dump(self.priority_clf, os.path.join(path, "priority_clf.joblib"))

    @classmethod
    def load(cls, path="checkpoints/baseline", taxonomy_path="config/taxonomy.json"):
        obj = cls(taxonomy_path)
        obj.vectorizer = joblib.load(os.path.join(path, "vectorizer.joblib"))
        obj.category_clf = joblib.load(os.path.join(path, "category_clf.joblib"))
        obj.subcategory_clf = joblib.load(os.path.join(path, "subcategory_clf.joblib"))
        obj.priority_clf = joblib.load(os.path.join(path, "priority_clf.joblib"))
        obj.is_fitted = True
        return obj
