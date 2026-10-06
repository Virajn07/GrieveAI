import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SyntheticDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "data/processed/grievances_synthetic.csv").open("r", encoding="utf-8") as f:
            cls.rows = list(csv.DictReader(f))
        cls.taxonomy = json.loads((ROOT / "config/taxonomy.json").read_text(encoding="utf-8"))

    def test_all_categories_present(self):
        self.assertEqual({r["category"] for r in self.rows}, set(self.taxonomy["categories"]))

    def test_subcategory_belongs_to_category(self):
        for row in self.rows:
            self.assertIn(row["subcategory"], self.taxonomy["categories"][row["category"]]["subcategories"])

    def test_priority_in_range(self):
        self.assertTrue(all(1 <= int(r["priority"]) <= 5 for r in self.rows))

    def test_three_demo_languages_present(self):
        self.assertEqual({r["language"] for r in self.rows}, {"en", "hi", "hinglish"})

    def test_duplicate_references_exist(self):
        ids = {r["id"] for r in self.rows}
        self.assertTrue(all(not r["duplicate_of"] or r["duplicate_of"] in ids for r in self.rows))


if __name__ == "__main__":
    unittest.main()
