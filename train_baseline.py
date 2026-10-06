"""
Trains the baseline TF-IDF + Logistic Regression classifier and saves it
to checkpoints/baseline/. Runs in seconds, no GPU needed.

Usage:  python train_baseline.py
The Flask app loads this checkpoint automatically on first request.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.baseline_classifier import BaselineGrievanceClassifier

df = pd.read_csv("data/processed/grievances_synthetic.csv")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["category"])

clf = BaselineGrievanceClassifier()
clf.fit(train_df["text"], train_df["category"], train_df["subcategory"], train_df["priority"])

preds = [clf.predict(t) for t in val_df["text"]]
cat_preds = [p["category"] for p in preds]
sub_preds = [p["subcategory"] for p in preds]
pri_preds = [str(p["priority"]) for p in preds]

print(f"Category accuracy:      {accuracy_score(val_df['category'], cat_preds):.3f}")
print(f"Category macro-F1:      {f1_score(val_df['category'], cat_preds, average='macro'):.3f}")
print(f"Sub-category accuracy:  {accuracy_score(val_df['subcategory'], sub_preds):.3f}")
print(f"Priority exact-match:   {accuracy_score(val_df['priority'].astype(str), pri_preds):.3f}")

clf.save()
print("\nSaved baseline model to checkpoints/baseline/")
