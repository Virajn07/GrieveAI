import unittest

import numpy as np

from src.priority_dedup import cluster_recurring_grievances, find_similar_grievances


class FixedTextEmbedder:
    vectors = {
        "wifi is not working in lab 3": [1.0, 0.0],
        "lab 3 wifi has stopped working": [0.99, 0.1],
        "no internet in lab 3": [0.97, 0.2],
        "exam hall ticket has a wrong center": [0.0, 1.0],
    }

    def encode(self, texts, normalize_embeddings=True):
        values = np.asarray([self.vectors[text.casefold()] for text in texts], dtype=float)
        if normalize_embeddings:
            values /= np.linalg.norm(values, axis=1, keepdims=True)
        return values


class SimilarityTests(unittest.TestCase):
    def setUp(self):
        self.embedder = FixedTextEmbedder()

    def test_returns_exact_near_duplicate_and_related_types(self):
        matches = find_similar_grievances(
            "WiFi is not working in Lab 3",
            [
                (1, "wifi is not working in lab 3", "IT_Library", "wifi_network"),
                (2, "Lab 3 wifi has stopped working", "IT_Library", "wifi_network"),
                (3, "no internet in lab 3", "IT_Library", "wifi_network"),
                (4, "exam hall ticket has a wrong center", "Examinations", "hall_ticket"),
            ],
            embedder=self.embedder,
            duplicate_threshold=0.99,
            related_threshold=0.75,
        )
        self.assertEqual([m["relationship"] for m in matches], ["exact_duplicate", "near_duplicate", "related"])
        self.assertEqual(matches[0]["grievance_id"], 1)
        self.assertGreater(matches[1]["similarity"], 0.95)

    def test_groups_recurring_issue_without_an_llm(self):
        groups = cluster_recurring_grievances(
            [
                (1, "wifi is not working in lab 3", "IT_Library", "wifi_network"),
                (2, "lab 3 wifi has stopped working", "IT_Library", "wifi_network"),
                (3, "no internet in lab 3", "IT_Library", "wifi_network"),
                (4, "exam hall ticket has a wrong center", "Examinations", "hall_ticket"),
            ],
            embedder=self.embedder,
            similarity_threshold=0.95,
            min_cluster_size=2,
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["grievance_count"], 3)
        self.assertEqual(groups[0]["issue_theme"], "IT_Library / wifi_network")
        self.assertEqual(groups[0]["grievance_ids"], ["1", "2", "3"])


if __name__ == "__main__":
    unittest.main()
