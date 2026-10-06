"""
Run:  python -m pytest tests/test_generate_data.py -v
(or just: python tests/test_generate_data.py)

Sanity-checks the synthetic dataset before it's used for training -
catches taxonomy/template mismatches early.
"""

import json
import os
import sys
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TAXONOMY_PATH = "config/taxonomy.json"
DATA_PATH = "data/processed/grievances_synthetic.csv"


def load_rows():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_all_categories_present():
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    rows = load_rows()
    seen_categories = {r["category"] for r in rows}
    expected = set(taxonomy["categories"].keys())
    assert seen_categories == expected, f"Missing categories: {expected - seen_categories}"


def test_subcategory_belongs_to_category():
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    rows = load_rows()
    for r in rows:
        valid_subs = taxonomy["categories"][r["category"]]["subcategories"]
        assert r["subcategory"] in valid_subs, \
            f"Row {r['id']}: {r['subcategory']} not valid under {r['category']}"


def test_priority_in_range():
    rows = load_rows()
    for r in rows:
        p = int(r["priority"])
        assert 1 <= p <= 5, f"Row {r['id']}: priority {p} out of range"


def test_all_three_languages_present():
    rows = load_rows()
    langs = {r["language"] for r in rows}
    assert langs == {"en", "hi", "hinglish"}, f"Unexpected languages: {langs}"


def test_duplicates_reference_valid_ids():
    rows = load_rows()
    ids = {r["id"] for r in rows}
    for r in rows:
        if r["duplicate_of"]:
            assert r["duplicate_of"] in ids, f"Row {r['id']} references missing id {r['duplicate_of']}"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\nAll {len(tests)} checks passed.")
