import json
import unittest
from pathlib import Path

import pandas as pd

from src.data_splits import has_group_leakage, make_splits, validate_taxonomy_labels


ROOT = Path(__file__).resolve().parents[1]


class SyntheticMLDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / "data/processed/grievances_synthetic.csv")
        cls.taxonomy = json.loads((ROOT / "config/taxonomy.json").read_text(encoding="utf-8"))

    def test_all_labels_match_taxonomy(self):
        validate_taxonomy_labels(self.data, self.taxonomy)
        self.assertEqual(self.data["category"].nunique(), 7)
        self.assertEqual(self.data["subcategory"].nunique(), 33)

    def test_invalid_category_subcategory_pair_is_rejected(self):
        changed = self.data.iloc[:2].copy()
        changed.iloc[0, changed.columns.get_loc("subcategory")] = "wifi_network"
        with self.assertRaisesRegex(ValueError, "category_subcategory_pairs"):
            validate_taxonomy_labels(changed, self.taxonomy)

    def test_taxonomy_rejects_subcategory_with_multiple_parents(self):
        changed = json.loads(json.dumps(self.taxonomy))
        changed["categories"]["Academics"]["subcategories"].append("wifi_network")
        with self.assertRaisesRegex(ValueError, "exactly one parent"):
            validate_taxonomy_labels(self.data, changed)

    def test_split_is_deterministic_and_has_no_duplicate_leakage(self):
        first = make_splits(self.data, seed=42)
        second = make_splits(self.data, seed=42)
        self.assertEqual([list(part.index) for part in first], [list(part.index) for part in second])
        self.assertFalse(has_group_leakage(first))
        self.assertEqual(tuple(len(part) for part in first), tuple(len(part) for part in second))


if __name__ == "__main__":
    unittest.main()
