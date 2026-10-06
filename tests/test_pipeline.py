"""Offline end-to-end checks using the checked-in synthetic baseline only."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from app.app import create_app
from app.models_db import AuditLog, Grievance, db
from app.routes import sla_status
from src.language_id import detect_language_details, redact_pii
from src.llm_report import summarize_and_route


class PipelineSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = None
        self.previous_token = os.environ.get("ADMIN_TOKEN")
        self.previous_dedupe_setting = os.environ.get("DISABLE_SEMANTIC_DEDUP")
        os.environ["ADMIN_TOKEN"] = "test-admin-token"
        os.environ["DISABLE_SEMANTIC_DEDUP"] = "1"
        self.app = create_app({
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite://",
            "CONFIDENCE_THRESHOLD": 1.0,
        })
        self.client = self.app.test_client()
        with self.app.app_context():
            connection = db.engine.connect()
            try:
                migration_config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
                migration_config.attributes["connection"] = connection
                command.upgrade(migration_config, "head")
            finally:
                connection.close()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
        if self.previous_token is None:
            os.environ.pop("ADMIN_TOKEN", None)
        else:
            os.environ["ADMIN_TOKEN"] = self.previous_token
        if self.previous_dedupe_setting is None:
            os.environ.pop("DISABLE_SEMANTIC_DEDUP", None)
        else:
            os.environ["DISABLE_SEMANTIC_DEDUP"] = self.previous_dedupe_setting

    def test_language_and_pii(self):
        self.assertEqual(detect_language_details("wifi nahi chal raha")["language"], "hinglish")
        self.assertEqual(detect_language_details("पानी की समस्या")["script"], "devanagari")
        clean = redact_pii("Contact me at student@example.edu or 9876543210, roll AB123456")
        self.assertNotIn("student@example.edu", clean)
        self.assertNotIn("9876543210", clean)
        self.assertNotIn("AB123456", clean)

    def test_full_pipeline_review_routing_tracking_audit_and_override(self):
        self.assertIn(self.client.get("/").status_code, (302, 303))
        login = self.client.post("/admin/login", data={"token": "test-admin-token"})
        self.assertIn(login.status_code, (302, 303))
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/submit").status_code, 200)
        health = self.client.get("/api/v1/health")
        self.assertEqual(health.status_code, 200)
        response = self.client.post("/api/v1/grievances", json={
            "text": "Library WiFi has not worked for three days. Call 9876543210, roll AB123456, mail student@example.edu"
        })
        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertRegex(payload["ack_number"], r"^GRV-\d{8}-[A-F0-9]{12}$")
        self.assertTrue(payload["manual_review"], "threshold 1.0 must send the case to review")
        self.assertIsNotNone(payload["category"])
        self.assertIsNotNone(payload["subcategory"])
        self.assertGreaterEqual(payload["priority"], 1)
        self.assertLessEqual(payload["priority"], 5)
        self.assertEqual(payload["script"], "latin")
        self.assertIsNone(payload["explanation"], "explanation should be generated only on admin request")

        ack = payload["ack_number"]
        inbox = self.client.get("/?status=review")
        self.assertEqual(inbox.status_code, 200)
        self.assertIn(ack, inbox.get_data(as_text=True))
        with self.app.app_context():
            row = Grievance.query.filter_by(ack_number=ack).one()
            self.assertIsNone(row.explanation)
            self.assertNotIn("9876543210", row.text)
            self.assertNotIn("student@example.edu", row.text)
            self.assertIsNotNone(row.duplicate_method)
            self.assertIsNotNone(row.manual_review_at)
        self.assertEqual(self.client.get("/api/v1/grievances/" + ack).status_code, 200)
        self.assertEqual(self.client.get(payload["track_url"]).status_code, 200)

        second = self.client.post("/api/v1/grievances", json={
            "text": "Library WiFi has not worked for three days. Call 9876543210, roll AB123456, mail student@example.edu"
        })
        self.assertEqual(second.status_code, 200, second.get_json())
        self.assertEqual(second.get_json()["duplicate_of"], 1)
        self.assertEqual(second.get_json()["related_matches"][0]["relationship"], "exact_duplicate")

        headers = {"X-ADMIN-TOKEN": "test-admin-token"}
        queue = self.client.get("/api/v1/review-queue", headers=headers)
        self.assertEqual(queue.status_code, 200)
        self.assertEqual(queue.get_json()["items"][0]["ack_number"], ack)
        bypass = self.client.post("/api/v1/grievances/" + ack + "/status", json={"status": "in_progress"}, headers=headers)
        self.assertEqual(bypass.status_code, 409)

        route = self.client.post("/api/v1/grievances/" + ack + "/override", json={
            "department": "IT Services / Library", "actor": "smoke-test", "note": "reviewed"
        }, headers=headers)
        self.assertEqual(route.status_code, 200, route.get_json())
        status = self.client.post("/api/v1/grievances/" + ack + "/status", json={"status": "in_progress"}, headers=headers)
        self.assertEqual(status.status_code, 200)
        with self.app.app_context():
            row = Grievance.query.filter_by(ack_number=ack).one()
            self.assertEqual(row.routed_department, "IT Services / Library")
            self.assertFalse(row.manual_review)
            self.assertTrue(AuditLog.query.filter_by(action="override").count())
        self.assertEqual(self.client.get("/admin/audit/" + ack).status_code, 200)
        labels = self.client.post("/api/v1/grievances/" + ack + "/classification", json={
            "category": "Canteen", "subcategory": "food_hygiene", "priority": 4, "actor": "smoke-test"
        }, headers=headers)
        self.assertEqual(labels.status_code, 200, labels.get_json())
        self.assertEqual(labels.get_json()["routed_department"], "Canteen Management")
        bad_labels = self.client.post("/api/v1/grievances/" + ack + "/classification", json={
            "category": "Canteen", "subcategory": "wifi_network", "priority": 4
        }, headers=headers)
        self.assertEqual(bad_labels.status_code, 400)
        summary_review = self.client.post("/api/v1/grievances/" + ack + "/summary-review", json={
            "factuality": "partially_factual", "note": "reviewed; contact alice@example.edu", "actor": "admin@example.edu"
        }, headers=headers)
        self.assertEqual(summary_review.status_code, 200)
        explanation = self.client.get("/api/v1/grievances/" + ack + "/explanation", headers=headers)
        self.assertTrue(explanation.get_json()["available"])
        self.assertTrue(explanation.get_json()["features"])
        with self.app.app_context():
            row = Grievance.query.filter_by(ack_number=ack).one()
            self.assertEqual(row.predicted_category, payload["category"])
            self.assertEqual(row.predicted_subcategory, payload["subcategory"])
            self.assertEqual(row.predicted_priority, payload["priority"])
            self.assertTrue(row.model_version)
            self.assertNotEqual(row.category, row.predicted_category)
            self.assertEqual(row.model_department, "IT Services / Library")
            self.assertNotIn("9876543210", str(row.explanation))
            self.assertEqual(row.explanation, explanation.get_json()["features"])
            second_row = Grievance.query.filter_by(ack_number=second.get_json()["ack_number"]).one()
            self.assertEqual(second_row.duplicate_of_id, row.id)
            self.assertIsNotNone(row.recurring_cluster_id)
            self.assertEqual(second_row.recurring_cluster_id, row.recurring_cluster_id)
            self.assertIsNotNone(second_row.duplicate_similarity)
            self.assertIsNotNone(row.sla_deadline)
            index_names = {index["name"] for index in db.inspect(db.engine).get_indexes("grievances")}
            expected_indexes = {"ix_grievances_routed_department", "ix_grievances_category", "ix_grievances_priority", "ix_grievances_sla_deadline", "ix_grievances_duplicate_of_id"}
            self.assertTrue(expected_indexes.issubset(index_names))
            self.assertTrue(AuditLog.query.filter_by(action="classification_override").count())
            event = AuditLog.query.filter_by(action="summary_review").one()
            self.assertNotIn("alice@example.edu", event.detail)
            self.assertNotIn("alice@example.edu", row.summary_review_note)
            self.assertNotIn("admin@example.edu", row.summary_reviewed_by)
        self.assertEqual(self.client.post("/api/v1/grievances/" + ack + "/status", json={"status": "submitted"}, headers=headers).status_code, 409)
        metrics = self.client.get("/api/v1/admin/metrics", headers=headers).get_json()
        self.assertIsNone(metrics["routing_accuracy"])
        self.assertEqual(self.app.test_client().get("/api/v1/admin/analytics").status_code, 401)
        analytics = self.client.get("/api/v1/admin/analytics", headers=headers)
        self.assertEqual(analytics.status_code, 200)
        self.assertEqual(analytics.get_json()["source"], "local synthetic/demo submissions; workflow counts are not model evaluation metrics")
        self.assertGreaterEqual(analytics.get_json()["duplicate_count"], 1)
        corpus = analytics.get_json()["synthetic_training_dataset"]
        self.assertTrue(corpus["available"])
        self.assertGreater(corpus["total_grievances"], 0)
        self.assertIn("no timestamp column", corpus["daily_trends_note"])
        with self.client.session_transaction() as browser_session:
            browser_session["admin_authenticated"] = True
        dashboard = self.client.get("/")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn(b"Synthetic training corpus", dashboard.data)

    def test_seven_categories_languages_and_sla(self):
        examples = [
            ("Sab departments ka lunch break same hai, canteen mein bohot crowd hota hai, break timing review karo.", "Academics", "hinglish"),
            ("the elective course ke internal aur external marks mein mismatch hai.", "Examinations", "hinglish"),
            ("मुझे स्कॉलरशिप एप्लीकेशन के बारे में कोई अपडेट नहीं मिला।", "Fees_Accounts", "hi"),
            ("I cannot access the online journal or digital library resource required for my course.", "IT_Library", "en"),
            ("There is not enough two-wheeler parking space for students on campus.", "Infrastructure", "en"),
            ("Boisar se daily travel ki wajah se students ke liye full college schedule follow karna difficult ho raha hai.", "Transport", "hinglish"),
            ("Canteen ka menu mahino se nahi badla, hume aur variety chahiye.", "Canteen", "hinglish"),
        ]
        ack_numbers = []
        for text, expected_category, expected_language in examples:
            response = self.client.post("/api/v1/grievances", json={"text": text})
            self.assertEqual(response.status_code, 200, response.get_json())
            body = response.get_json()
            self.assertEqual(body["category"], expected_category)
            self.assertEqual(body["language"], expected_language)
            self.assertTrue(body["manual_review"])
            ack_numbers.append(body["ack_number"])
        self.assertEqual(len(set(ack_numbers)), 7)
        with self.app.app_context():
            rows = Grievance.query.filter(Grievance.ack_number.in_(ack_numbers)).all()
            self.assertEqual(len(rows), 7)
            self.assertTrue(all(row.sla_deadline is not None for row in rows))
            self.assertEqual(sla_status(rows[0]), "on_track")
            future = rows[0].sla_deadline.replace(year=rows[0].sla_deadline.year + 1)
            self.assertEqual(sla_status(rows[0], at=future), "overdue")

    def test_high_confidence_automatic_routing(self):
        class KnownTransportClassifier:
            confidence_threshold = 0.55
            model_version = "test_transport_v1"
            vectorizer = None

            def predict(self, _text):
                return {
                    "category": "Transport", "subcategory": "commuting_accessibility",
                    "priority": 3, "priority_raw": 3.0, "confidence": 0.99,
                    "subcategory_confidence": 0.9,
                }

            def explain(self, _text):
                return []

        self.app.config["CONFIDENCE_THRESHOLD"] = 0.0
        with patch("app.services.get_classifier", return_value=KnownTransportClassifier()):
            response = self.client.post("/api/v1/grievances", json={
                "text": "Synthetic transport routing check: the student bus route is delayed."
            })
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertFalse(body["manual_review"])
        self.assertEqual(body["routed_department"], "Transport Committee")
        ticket = self.client.get("/api/v1/grievances/" + body["ack_number"]).get_json()
        self.assertEqual(ticket["status"], "routed")
        self.assertEqual(ticket["sla_status"], "on_track")

    def test_admin_fails_closed_and_invalid_input(self):
        self.assertEqual(self.client.get("/api/v1/review-queue").status_code, 401)
        self.assertEqual(self.client.post("/api/v1/grievances", json={"text": "  "}).status_code, 400)
        self.assertEqual(self.client.post("/api/v1/grievances", json={"text": "x" * 2001}).status_code, 400)

    def test_admin_disabled_without_configured_token(self):
        os.environ.pop("ADMIN_TOKEN", None)
        try:
            self.assertEqual(self.client.get("/api/v1/review-queue").status_code, 503)
            self.assertEqual(self.client.post("/admin/login", data={"token": "anything"}).status_code, 503)
        finally:
            os.environ["ADMIN_TOKEN"] = "test-admin-token"

    def test_summary_fallback_and_configured_route_without_credentials(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "OPENROUTER_API_KEY": ""}, clear=False):
            result = summarize_and_route("wifi nahi chal raha", "IT_Library", "wifi_network")
        self.assertEqual(result["provider"], "deterministic")
        self.assertEqual(result["recommended_department"], "IT Services / Library")


if __name__ == "__main__":
    unittest.main()
