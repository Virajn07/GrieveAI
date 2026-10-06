import os
import unittest
from unittest.mock import patch

from src.llm_report import summarize_and_route


class StructuredLlmTests(unittest.TestCase):
    def setUp(self):
        self.prediction = {
            "category": "IT_Library", "subcategory": "wifi_network", "priority": 3, "confidence": 0.82
        }
        self.valid = {
            "summary": "WiFi is unavailable in Lab 3.",
            "root_cause": "Not established from the report.",
            "recommended_action": "Check the Lab 3 access point and report findings.",
            "department_recommendation": "IT Services / Library",
            "urgency_reason": "The issue affects access to campus internet.",
            "recurring_issue": True,
            "recurring_interpretation": "Several similar lab connectivity reports were found.",
        }

    def test_no_key_returns_valid_deterministic_structure(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "OPENROUTER_MODEL": ""}):
            result = summarize_and_route("Lab WiFi is down", **{
                "category": self.prediction["category"],
                "subcategory": self.prediction["subcategory"],
                "priority": self.prediction["priority"],
                "confidence": self.prediction["confidence"],
            })
        self.assertEqual(result["provider"], "deterministic")
        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["recommended_department"], "IT Services / Library")
        self.assertEqual(result["department_recommendation"], "IT Services / Library")
        self.assertIn("root_cause", result)

    def test_valid_openrouter_json_is_used_but_does_not_override_routing(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "unit-test", "OPENROUTER_MODEL": "test-model"}):
            with patch("src.llm_report._request_openrouter", return_value=self.valid):
                result = summarize_and_route(
                    "Lab WiFi is down", "IT_Library", "wifi_network", priority=3, confidence=0.82
                )
        self.assertEqual(result["provider"], "openrouter")
        self.assertFalse(result["used_fallback"])
        self.assertEqual(result["recommended_department"], "IT Services / Library")
        self.assertEqual(result["summary"], self.valid["summary"])

    def test_invalid_json_structure_falls_back(self):
        invalid = dict(self.valid, department_recommendation="Unconfigured Department")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "unit-test", "OPENROUTER_MODEL": "test-model"}):
            with self.assertLogs("src.llm_report", level="WARNING"):
                with patch("src.llm_report._request_openrouter", return_value=invalid):
                    result = summarize_and_route("Lab WiFi is down", "IT_Library", "wifi_network")
        self.assertEqual(result["provider"], "deterministic")
        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["recommended_department"], "IT Services / Library")


if __name__ == "__main__":
    unittest.main()
