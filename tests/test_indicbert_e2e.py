"""Opt-in integration test using the real IndicBERT checkpoint and base model."""

import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config

from app.app import create_app
from app.models_db import AuditLog, Grievance, db
from src.inference import reset_classifier_cache


@unittest.skipUnless(
    os.getenv("GRIEVEAI_RUN_MODEL_INTEGRATION") == "1",
    "set GRIEVEAI_RUN_MODEL_INTEGRATION=1 to run real IndicBERT integration",
)
class IndicBERTEndToEndTests(unittest.TestCase):
    def test_seven_categories_persist_restart_correct_and_render(self):
        checkpoint = Path(os.getenv("INDICBERT_CHECKPOINT_DIR", "checkpoints/indicbert_lora/run1"))
        base_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--ai4bharat--IndicBERTv2-MLM-only"
        if not (checkpoint / "heads.pt").is_file() or not base_cache.exists():
            self.skipTest("real IndicBERT checkpoint or cached base model is unavailable")

        examples = [
            "Sir meri attendance 68% hai aur teacher ne mujhe exam ke liye allow nahi kiya.",
            "I cannot download my examination hall ticket even though my form was approved.",
            "मेरी छात्रवृत्ति की फीस अभी तक खाते में जमा नहीं हुई है।",
            "The library Wi-Fi has been unavailable since yesterday and students cannot access journals.",
            "Lab 3 lights are flickering and two electrical sockets are broken.",
            "Boisar ki last bus leaves before evening classes finish, please review the route timing.",
            "कैंटीन में खाना ठंडा मिलता है और साफ-सफाई भी ठीक नहीं है।",
        ]
        database_path = Path("app/instance") / f"indicbert-e2e-{uuid.uuid4().hex}.sqlite3"
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: database_path.unlink(missing_ok=True))
        database_url = f"sqlite:///{database_path.resolve()}"
        env = {
            "MODEL_BACKEND": "indicbert",
            "INDICBERT_CHECKPOINT_DIR": str(checkpoint),
            "TAXONOMY_CONFIG": "config/taxonomy.json",
            "DATABASE_URL": database_url,
            "ADMIN_TOKEN": "integration-admin-token",
            "ML_CONFIDENCE_THRESHOLD": "0.70",
            "OPENROUTER_API_KEY": "",
            "OPENROUTER_MODEL": "",
            "HF_HUB_OFFLINE": "1",
            "DISABLE_SEMANTIC_DEDUP": "1",
        }
        reset_classifier_cache()
        with patch.dict(os.environ, env):
            app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": database_url})
            self.addCleanup(self._close_database, app)
            with app.app_context():
                connection = db.engine.connect()
                try:
                    migration = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
                    migration.attributes["connection"] = connection
                    command.upgrade(migration, "head")
                finally:
                    connection.close()

            client = app.test_client()
            readiness = client.get("/api/v1/health")
            self.assertEqual(readiness.status_code, 200, readiness.get_json())
            self.assertTrue(readiness.get_json()["model"]["version"].startswith("indicbert_lora_run1+base-"))
            created = []
            expected_categories = [
                "Academics", "Examinations", "Fees_Accounts", "IT_Library",
                "Infrastructure", "Transport", "Canteen",
            ]
            for text, expected_category in zip(examples, expected_categories):
                response = client.post("/api/v1/grievances", json={"text": text})
                self.assertEqual(response.status_code, 200, response.get_json())
                payload = response.get_json()
                self.assertTrue(payload["model_version"].startswith("indicbert_lora_run1+base-"))
                self.assertEqual(payload["category"], expected_category)
                self.assertGreaterEqual(payload["priority"], 1)
                self.assertLessEqual(payload["priority"], 5)
                self.assertIsInstance(payload["confidence"], float)
                self.assertIsInstance(payload["manual_review"], bool)
                self.assertEqual(payload["manual_review"], payload["confidence"] < 0.70)
                created.append(payload)

            self.assertTrue({"en", "hi", "hinglish"}.issubset({item["language"] for item in created}))
            tracked = client.get(f"/api/v1/grievances/{created[0]['ack_number']}")
            self.assertEqual(tracked.status_code, 200)
            self.assertEqual(tracked.get_json()["predicted_category"], created[0]["category"])
            self.assertEqual(tracked.get_json()["model_version"], created[0]["model_version"])
            explanation = client.get(
                f"/api/v1/grievances/{created[0]['ack_number']}/explanation",
                headers={"X-ADMIN-TOKEN": "integration-admin-token"},
            )
            self.assertEqual(explanation.status_code, 200, explanation.get_json())
            self.assertTrue(explanation.get_json()["available"])
            self.assertTrue(explanation.get_json()["features"])

            duplicate_response = client.post(
                "/api/v1/grievances", json={"text": examples[0]}
            )
            self.assertEqual(duplicate_response.status_code, 200, duplicate_response.get_json())
            duplicate = duplicate_response.get_json()
            self.assertIsNotNone(duplicate["duplicate_of"])
            self.assertIsNotNone(duplicate["recurring_cluster_id"])

            with app.app_context():
                rows = Grievance.query.order_by(Grievance.id).all()
                self.assertEqual(len(rows), 8)
                self.assertTrue(all(row.model_version.startswith("indicbert_lora_run1+base-") for row in rows))
                self.assertEqual(rows[0].predicted_category, created[0]["category"])
                original_prediction = (
                    rows[0].predicted_category,
                    rows[0].predicted_subcategory,
                    rows[0].predicted_priority,
                )
                ack = rows[0].ack_number

            correction_category = "Infrastructure" if original_prediction[0] != "Infrastructure" else "Transport"
            correction_subcategory = (
                "classroom_lab_maintenance" if correction_category == "Infrastructure" else "commuting_accessibility"
            )
            correction = client.post(
                f"/api/v1/grievances/{ack}/classification",
                json={
                    "category": correction_category,
                    "subcategory": correction_subcategory,
                    "priority": 3,
                    "actor": "reviewer",
                },
                headers={"X-ADMIN-TOKEN": "integration-admin-token"},
            )
            self.assertEqual(correction.status_code, 200, correction.get_json())

            app2 = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": database_url})
            self.addCleanup(self._close_database, app2)
            with app2.app_context():
                persisted = Grievance.query.filter_by(ack_number=ack).one()
                self.assertEqual(
                    (persisted.predicted_category, persisted.predicted_subcategory, persisted.predicted_priority),
                    original_prediction,
                )
                self.assertEqual(persisted.category, correction_category)
                self.assertTrue(AuditLog.query.filter_by(grievance_id=persisted.id, action="classification_override").count())
            with app2.test_client() as client2:
                with client2.session_transaction() as session:
                    session["admin_authenticated"] = True
                dashboard = client2.get("/")
                self.assertEqual(dashboard.status_code, 200)
                self.assertIn(b"indicbert_lora_run1+base-", dashboard.data)
                self.assertIn(b"Recurring group", dashboard.data)
                self.assertIn(b"Department suggestion", dashboard.data)

        reset_classifier_cache()

    @staticmethod
    def _close_database(app):
        with app.app_context():
            db.session.remove()
            db.engine.dispose()


if __name__ == "__main__":
    unittest.main()
