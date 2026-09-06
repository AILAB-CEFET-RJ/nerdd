import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.ner_score_calibrator import (  # noqa: E402
    apply_calibrator_to_score,
    build_examples_from_oof_rows,
    fit_ner_score_calibrator,
    metric_summary,
)
from calibration.serialization import apply_calibrator_to_score as apply_serialized_calibrator_to_score  # noqa: E402
from tools.apply_ner_score_calibrator import calibrated_rows  # noqa: E402


class NerScoreCalibratorTests(unittest.TestCase):
    def test_build_examples_marks_exact_prediction_as_positive(self):
        rows = [
            {
                "text": "fato na Rua Alfa",
                "gold_spans": [{"start": 8, "end": 16, "label": "Location"}],
                "pred_spans": [
                    {"start": 8, "end": 16, "label": "Location", "score": 0.9},
                    {"start": 8, "end": 12, "label": "Location", "score": 0.8},
                    {"start": 8, "end": 16, "label": "Organization", "score": 0.7},
                ],
            }
        ]

        examples, support = build_examples_from_oof_rows(
            rows,
            labels={"Location", "Organization"},
        )

        self.assertEqual(support, {"Location": 1})
        self.assertEqual([row["target"] for row in examples], [1, 0, 0])

    def test_fit_uses_identity_when_label_has_too_few_examples(self):
        examples = [
            {"label": "Location", "score_raw": 0.2, "target": 0, "row_index_1based": 1, "start": 0, "end": 1},
            {"label": "Location", "score_raw": 0.9, "target": 1, "row_index_1based": 1, "start": 2, "end": 3},
        ]

        calibrator, rows = fit_ner_score_calibrator(
            examples,
            labels=["Location"],
            method="isotonic",
            min_positive=2,
            min_negative=2,
        )

        self.assertEqual(calibrator["label_models"]["Location"]["method"], "identity")
        self.assertEqual([row["score_calibrated"] for row in rows], [0.2, 0.9])

    def test_fit_isotonic_and_apply_per_label(self):
        examples = []
        for i in range(25):
            examples.append(
                {"label": "Location", "score_raw": 0.1 + i * 0.005, "target": 0, "row_index_1based": i, "start": 0, "end": 1}
            )
            examples.append(
                {"label": "Location", "score_raw": 0.8 + i * 0.005, "target": 1, "row_index_1based": i, "start": 2, "end": 3}
            )

        calibrator, _rows = fit_ner_score_calibrator(
            examples,
            labels=["Location"],
            method="isotonic",
            min_positive=20,
            min_negative=20,
        )

        self.assertEqual(calibrator["label_models"]["Location"]["method"], "isotonic")
        low = apply_calibrator_to_score(0.15, "Location", calibrator)
        high = apply_calibrator_to_score(0.9, "Location", calibrator)
        self.assertLess(low, high)

    def test_metric_summary_reports_brier_scores(self):
        rows = [
            {"label": "Location", "score_raw": 0.9, "score_calibrated": 0.8, "target": 1},
            {"label": "Location", "score_raw": 0.8, "score_calibrated": 0.2, "target": 0},
        ]

        summary, raw_bins, calibrated_bins = metric_summary(rows, labels=["Location"], bins=2)

        self.assertIn("ALL", summary)
        self.assertIn("raw_brier", summary["Location"])
        self.assertEqual(len(raw_bins), 4)
        self.assertEqual(len(calibrated_bins), 4)

    def test_apply_script_helper_adds_calibrated_score(self):
        calibrator = {
            "kind": "ner_score_calibrator_oof",
            "label_models": {
                "Location": {"method": "isotonic", "x_thresholds": [0.0, 1.0], "y_thresholds": [0.0, 0.5]}
            },
            "fallback_model": {"method": "identity"},
        }
        rows = [{"entities": [{"text": "Rua Alfa", "label": "Location", "score": 0.8}]}]

        output, stats = calibrated_rows(
            rows,
            calibrator=calibrator,
            entity_key="entities",
            fallback_entity_key="ner",
            score_field="score",
            output_score_field="score_calibrated",
        )

        self.assertAlmostEqual(output[0]["entities"][0]["score_calibrated"], 0.4)
        self.assertEqual(stats["entities_calibrated"], 1)

    def test_serialization_apply_supports_ner_oof_calibrator(self):
        calibrator = {
            "kind": "ner_score_calibrator_oof",
            "label_models": {
                "Location": {"method": "isotonic", "x_thresholds": [0.0, 1.0], "y_thresholds": [0.0, 0.5]}
            },
            "fallback_model": {"method": "identity"},
        }

        self.assertAlmostEqual(
            apply_serialized_calibrator_to_score(0.8, "Location", calibrator),
            0.4,
        )


if __name__ == "__main__":
    unittest.main()
