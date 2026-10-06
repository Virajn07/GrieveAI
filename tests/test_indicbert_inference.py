"""Offline contract checks for the real IndicBERT checkpoint adapter."""

import unittest
from unittest.mock import patch

import torch

from src.indicbert_inference import IndicBERTInference, _taxonomy


class IndicBERTInferenceContractTests(unittest.TestCase):
    def test_hierarchical_prediction_and_model_identity(self):
        model = IndicBERTInference.__new__(IndicBERTInference)
        model.taxonomy = _taxonomy("config/taxonomy.json")
        model.model_version = "indicbert_lora_run1"
        category_logits = torch.tensor([[8.0, 0, 0, 0, 0, 0, 0]])
        subcategory_logits = torch.zeros((1, 33))
        subcategory_logits[0, 5] = 100.0  # invalid: belongs to Examinations
        subcategory_logits[0, 4] = 2.0  # valid: Academics/attendance
        output = {
            "category_logits": category_logits,
            "subcategory_logits": subcategory_logits,
            "priority_pred": torch.tensor([2.6]),
        }
        with patch.object(model, "_forward", return_value=output):
            result = model.predict("मेरी attendance problem है")
        self.assertEqual(result["category"], "Academics")
        self.assertEqual(result["subcategory"], "attendance")
        self.assertEqual(result["priority"], 3)
        self.assertEqual(result["model_version"], "indicbert_lora_run1")
        self.assertGreater(result["confidence"], 0.9)

    def test_missing_checkpoint_fails_with_actionable_message(self):
        with self.assertRaisesRegex(RuntimeError, "IndicBERT checkpoint is incomplete"):
            IndicBERTInference("missing/checkpoint", "config/taxonomy.json")


if __name__ == "__main__":
    unittest.main()
