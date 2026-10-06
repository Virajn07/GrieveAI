import unittest
from types import SimpleNamespace

import numpy as np

from src.explainability import explain_transformer


class ExplainabilityTests(unittest.TestCase):
    def test_ranks_tokens_for_predicted_class(self):
        class Explainer:
            def __call__(self, _texts):
                return SimpleNamespace(
                    data=[["attendance", "exam", "allowed"]],
                    values=np.asarray([[[0.1, 0.8], [0.5, -0.2], [0.0, 0.3]]]),
                )

        result = explain_transformer(Explainer(), "attendance exam allowed", predicted_class_idx=1, top_k=2)
        self.assertEqual(result, [("attendance", 0.8), ("allowed", 0.3)])


if __name__ == "__main__":
    unittest.main()
