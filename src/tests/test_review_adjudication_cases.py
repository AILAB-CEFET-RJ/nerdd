import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.review_adjudication_cases import render_adjudication_review


class ReviewAdjudicationCasesTests(unittest.TestCase):
    def test_render_adjudication_review_includes_multiple_layers_and_scores(self):
        rows = [
            {
                "source_id": "candidate_rank_1",
                "adjudication_priority_score": 0.91,
                "_source": {
                    "source_id": "candidate_rank_1",
                    "text": "Rua A em Belford Roxo",
                    "baseline_entities": [
                        {"text": "Rua A", "label": "Location", "start": 0, "end": 5, "ner_score": 0.81}
                    ],
                    "gliner2_entities": [
                        {"text": "Belford Roxo", "label": "Location", "start": 9, "end": 21, "confidence": 0.92}
                    ],
                    "review_seed_entities": [
                        {"text": "Rua A", "label": "Location", "start": 0, "end": 5, "ner_score": 0.81},
                        {"text": "Belford Roxo", "label": "Location", "start": 9, "end": 21, "confidence": 0.92},
                    ],
                },
                "adjudication": {
                    "decision": "accept_with_edits",
                    "review_confidence": "high",
                    "entities_final": [
                        {"text": "Rua A", "label": "Location", "start": 0, "end": 5},
                        {"text": "Belford Roxo", "label": "Location", "start": 9, "end": 21},
                    ],
                },
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "review.html"
            render_adjudication_review(
                rows,
                output_path=str(output),
                title="Adjudication Review",
                layers=["baseline_entities", "gliner2_entities", "review_seed_entities", "adjudication.entities_final"],
                score_fields=["ner_score", "confidence"],
            )
            html = output.read_text(encoding="utf-8")

        self.assertIn("Baseline", html)
        self.assertIn("GLiNER2", html)
        self.assertIn("Review Seeds", html)
        self.assertIn("Adjudicated Final", html)
        self.assertIn("candidate_rank_1", html)
        self.assertIn("accept_with_edits", html)
        self.assertIn("[0.810]", html)
        self.assertIn("[0.920]", html)


if __name__ == "__main__":
    unittest.main()
