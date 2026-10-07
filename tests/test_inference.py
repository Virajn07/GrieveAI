import unittest
from unittest.mock import patch

from src.baseline_classifier import BaselineGrievanceClassifier
from src.inference import predict_grievance
from src.configuration import load_threshold_settings
from src.inference import get_classifier, reset_classifier_cache
from src.language_id import redact_pii


class InferenceApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = BaselineGrievanceClassifier.load()

    def test_returns_redacted_structured_prediction_and_model_version(self):
        result = predict_grievance(
            "Meri attendance kam hai. Contact student@example.edu, roll AB123456, phone 9876543210.",
            classifier=self.classifier,
            confidence_threshold=1.0,
        )
        self.assertEqual(result["language"], "hinglish")
        self.assertIn(result["category"], self.classifier.taxonomy["categories"])
        self.assertIn(
            result["subcategory"],
            self.classifier.taxonomy["categories"][result["category"]]["subcategories"],
        )
        self.assertGreaterEqual(result["priority"], 1)
        self.assertLessEqual(result["priority"], 5)
        self.assertTrue(result["requires_human_review"])
        self.assertEqual(result["confidence_threshold"], 1.0)
        self.assertEqual(result["model_version"], self.classifier.model_version)
        for identifier in ("student@example.edu", "AB123456", "9876543210"):
            self.assertNotIn(identifier, result["redacted_text"])

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            predict_grievance("  ", classifier=self.classifier)

    def test_redacts_common_indian_identifier_formats(self):
        text = "Call 98765 43210, roll AB123456, student ID 23BCE1234, Aadhaar 1234 5678 9012"
        redacted = redact_pii(text)
        for identifier in ("98765 43210", "AB123456", "23BCE1234", "1234 5678 9012"):
            self.assertNotIn(identifier, redacted)
        self.assertIn("roll call", redact_pii("The roll call begins at 10."))

    def test_redacts_explicit_english_and_hinglish_names(self):
        english = redact_pii(
            "My name is Rahul Sharma, my phone number is 9876543210 and WiFi keeps disconnecting."
        )
        hinglish = redact_pii(
            "Mera naam Test Student hai, student ID 24CS999 hai aur hostel WiFi disconnect hota hai."
        )
        for identifier in ("Rahul Sharma", "9876543210", "Test Student", "24CS999"):
            self.assertNotIn(identifier, english + " " + hinglish)
        self.assertIn("[NAME]", english)
        self.assertIn("[NAME]", hinglish)

    def test_confidence_threshold_is_configurable_by_environment(self):
        with patch.dict("os.environ", {"ML_CONFIDENCE_THRESHOLD": "0.72", "CONFIDENCE_THRESHOLD": "0.8"}):
            self.assertEqual(load_threshold_settings()["confidence_threshold"], 0.72)

    def test_indicbert_missing_checkpoint_does_not_silently_fall_back(self):
        reset_classifier_cache()
        env = {
            "MODEL_BACKEND": "indicbert",
            "INDICBERT_CHECKPOINT_DIR": "missing/indicbert-checkpoint-for-test",
            "TAXONOMY_CONFIG": "config/taxonomy.json",
        }
        try:
            with patch.dict("os.environ", env):
                with self.assertRaisesRegex(RuntimeError, "IndicBERT checkpoint is incomplete"):
                    get_classifier()
        finally:
            reset_classifier_cache()


if __name__ == "__main__":
    unittest.main()
