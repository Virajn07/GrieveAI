"""Shared validation and deterministic duplicate-aware data splits."""

from __future__ import annotations

import hashlib

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def validate_taxonomy_labels(df, taxonomy):
    """Fail early when dataset category/subcategory labels disagree with taxonomy."""
    required = {"category", "subcategory"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")

    categories = taxonomy["categories"]
    parent_by_subcategory = {}
    duplicate_subcategories = []
    for category, details in categories.items():
        for subcategory in details.get("subcategories", []):
            if subcategory in parent_by_subcategory:
                duplicate_subcategories.append(subcategory)
            else:
                parent_by_subcategory[subcategory] = category
    if duplicate_subcategories:
        raise ValueError(
            "Taxonomy subcategories must have exactly one parent: "
            f"{sorted(set(duplicate_subcategories))}"
        )

    invalid_categories = sorted(set(df["category"].astype(str)) - set(categories))
    invalid_pairs = sorted({
        (str(row.category), str(row.subcategory))
        for row in df[["category", "subcategory"]].itertuples(index=False)
        if str(row.category) not in categories
        or str(row.subcategory) not in categories[str(row.category)]["subcategories"]
    })
    if invalid_categories or invalid_pairs:
        raise ValueError(
            f"Dataset labels do not match taxonomy: categories={invalid_categories}, "
            f"category_subcategory_pairs={invalid_pairs[:10]}"
        )


def _groups(df):
    """Join duplicate references and exact normalized-text copies into groups."""
    ids = df["id"].astype(str).tolist()
    parents = {row_id: row_id for row_id in ids}

    def find(value):
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left, right):
        if left in parents and right in parents:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

    text_owner = {}
    for row in df.itertuples(index=False):
        row_id = str(getattr(row, "id"))
        duplicate_of = getattr(row, "duplicate_of", None)
        if pd.notna(duplicate_of) and str(duplicate_of).strip():
            union(row_id, str(duplicate_of).strip())
        normalized = " ".join(str(getattr(row, "text")).lower().split())
        text_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        if text_hash in text_owner:
            union(row_id, text_owner[text_hash])
        else:
            text_owner[text_hash] = row_id
    return pd.Series([find(row_id) for row_id in ids], index=df.index)


def make_splits(df, seed=42):
    """Return deterministic 60/20/20 train/validation/test partitions."""
    required = {"id", "text", "category", "subcategory", "priority", "duplicate_of"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    groups = _groups(df)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(outer.split(df, df["category"], groups))
    train_val = df.iloc[train_val_idx]
    inner_groups = groups.iloc[train_val_idx]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_rel, val_rel = next(inner.split(train_val, train_val["category"], inner_groups))
    return tuple(part.copy() for part in (
        train_val.iloc[train_rel], train_val.iloc[val_rel], df.iloc[test_idx]
    ))


def has_group_leakage(parts):
    """Check duplicate references and exact text copies across split boundaries."""
    ids = [set(part["id"].astype(str)) for part in parts]
    references = [
        {str(value).strip() for value in part["duplicate_of"].dropna() if str(value).strip()}
        for part in parts
    ]
    texts = [
        {" ".join(str(value).lower().split()) for value in part["text"]}
        for part in parts
    ]
    for left in range(len(parts)):
        for right in range(left + 1, len(parts)):
            if ids[left] & ids[right] or references[left] & ids[right] or references[right] & ids[left]:
                return True
            if texts[left] & texts[right]:
                return True
    return False
