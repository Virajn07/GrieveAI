import os
import json
import unittest
from urllib.error import HTTPError
from unittest.mock import Mock, patch

from src.llm_report import _request_ollama, summarize_and_route


class StructuredLlmTests(unittest.TestCase):
    def setUp(self):
        self.prediction = {
            "category": "IT_Library", "subcategory": "wifi_network", "priority": 3, "confidence": 0.82
        }
        self.valid = {
            "summary": "WiFi is unavailable in Lab 3.",
            "root_cause": "Not established from the report.",
            "recommended_action": "Check the Lab 3 access point and report findings.",
            "department": "IT Services / Library",
            "urgency_reason": "The issue affects access to campus internet.",
            "recurring_issue": True,
            "recurring_interpretation": "Several similar lab connectivity reports were found.",
        }

    def test_no_key_returns_valid_deterministic_structure(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "OPENROUTER_MODEL": ""}):
            with patch("src.llm_report._request_ollama", side_effect=ConnectionError("offline")):
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
            with patch("src.llm_report._request_ollama", side_effect=ConnectionError("offline")):
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
                with patch("src.llm_report._request_ollama", side_effect=ConnectionError("offline")):
                    with patch("src.llm_report._request_openrouter", return_value=invalid):
                        result = summarize_and_route("Lab WiFi is down", "IT_Library", "wifi_network")
        self.assertEqual(result["provider"], "deterministic")
        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["recommended_department"], "IT Services / Library")

    def test_provider_failure_does_not_break_fallback(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "unit-test", "OPENROUTER_MODEL": "test-model"}):
            with self.assertLogs("src.llm_report", level="WARNING"):
                with patch("src.llm_report._request_ollama", side_effect=ConnectionError("offline")):
                    with patch("src.llm_report._request_openrouter", side_effect=TimeoutError("mock timeout")) as request:
                        result = summarize_and_route(
                            "Lab WiFi is down", "IT_Library", "wifi_network", priority=3, confidence=0.82
                        )
        self.assertEqual(result["provider"], "deterministic")
        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["department_recommendation"], "IT Services / Library")
        self.assertEqual(result["fallback_label"], "Deterministic fallback — LLM unavailable")
        self.assertEqual([call.args[-1] for call in request.call_args_list], [
            "test-model", "apodex/apodex-1.1-mini:free"
        ])

    def test_backup_provider_is_used_after_primary_failure(self):
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "unit-test",
            "OPENROUTER_MODEL": "google/gemma-4-26b-a4b-it:free",
        }):
            with patch("src.llm_report._request_ollama", side_effect=ConnectionError("offline")):
                with patch("src.llm_report._request_openrouter", side_effect=[TimeoutError("primary"), self.valid]) as request:
                    result = summarize_and_route("Lab WiFi is down", "IT_Library", "wifi_network")
        self.assertEqual(request.call_args_list[0].args[-1], "google/gemma-4-26b-a4b-it:free")
        self.assertEqual(request.call_args_list[1].args[-1], "apodex/apodex-1.1-mini:free")
        self.assertEqual(result["provider_label"], "Apodex")
        self.assertEqual(result["model_name"], "apodex/apodex-1.1-mini:free")
        self.assertFalse(result["used_fallback"])

    def test_ollama_defaults_endpoint_model_and_redacts_all_outbound_grievance_text(self):
        body = {"message": {"content": json.dumps(self.valid)}}
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps(body).encode()
        raw = (
            "My name is Rahul Sharma, email rahul.test@example.com, phone 9876543210, "
            "student ID 24CS999. WiFi keeps disconnecting."
        )
        duplicate = [{"text": "same issue, student@example.edu, 9876543210, roll AB123456"}]
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "", "OLLAMA_MODEL": "", "LLM_TIMEOUT_SECONDS": "7"}):
            with patch("src.llm_report.urlopen", return_value=response) as urlopen:
                result = _request_ollama(raw, self.prediction, duplicate, [], {"IT_Library": "IT Services / Library"})
        request = urlopen.call_args.args[0]
        sent = json.loads(request.data.decode())
        context = json.loads(sent["messages"][1]["content"])
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(sent["model"], "qwen3:8b")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 7.0)
        for identifier in (
            "Rahul Sharma", "rahul.test@example.com", "9876543210", "24CS999",
            "student@example.edu", "AB123456",
        ):
            self.assertNotIn(identifier, json.dumps(context))
        self.assertEqual(result["summary"], self.valid["summary"])

    def test_successful_ollama_response_is_primary_provider(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "OLLAMA_MODEL": "qwen3:8b"}):
            with patch("src.llm_report._request_ollama", return_value=self.valid) as ollama:
                result = summarize_and_route("Lab WiFi is down", "IT_Library", "wifi_network")
        self.assertEqual(ollama.call_count, 1)
        self.assertEqual(result["provider"], "ollama")
        self.assertEqual(result["provider_label"], "Ollama")
        self.assertEqual(result["model_name"], "qwen3:8b")
        self.assertFalse(result["used_fallback"])

    def test_ollama_connection_failure_timeout_invalid_json_and_invalid_department_fall_back(self):
        failures = [
            ConnectionError("connection refused"),
            TimeoutError("request timed out"),
            HTTPError("http://127.0.0.1:11434/api/chat", 503, "unavailable", {}, None),
            ValueError("LLM response did not contain a valid JSON object"),
            dict(self.valid, department="Unconfigured Department"),
        ]
        for failure in failures:
            with self.subTest(failure=failure):
                with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
                    with patch("src.llm_report._request_ollama", side_effect=failure):
                        result = summarize_and_route("Lab WiFi is down", "IT_Library", "wifi_network")
                self.assertEqual(result["provider"], "deterministic")
                self.assertTrue(result["used_fallback"])

    def test_malformed_ollama_content_is_parsed_as_invalid_and_falls_back(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps({"message": {"content": "thinking... no JSON here"}}).encode()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            with patch("src.llm_report.urlopen", return_value=response):
                result = summarize_and_route("Lab WiFi is down", "IT_Library", "wifi_network")
        self.assertEqual(result["provider"], "deterministic")
        self.assertTrue(result["used_fallback"])


if __name__ == "__main__":
    unittest.main()
