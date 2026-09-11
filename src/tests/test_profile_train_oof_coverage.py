import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.profile_train_oof_coverage import build_profile


def _span(text, mention, label):
    start = text.index(mention)
    return {"start": start, "end": start + len(mention), "label": label}


class ProfileTrainOofCoverageTests(unittest.TestCase):
    def test_profiles_strict_oof_outcomes_and_train_coverage(self):
        train_one = "Tráfico na Rua Alfa com João"
        train_two = "Rua Alfa"
        train_three = "Bairro Beta"
        train_rows = [
            {"text": train_one, "spans": [_span(train_one, "Rua Alfa", "Location"), _span(train_one, "João", "Person")]},
            {"text": train_two, "spans": [_span(train_two, "Rua Alfa", "Location")]},
            {"text": train_three, "spans": [_span(train_three, "Bairro Beta", "Location")]},
        ]

        false_positive_text = "Rua Gama"
        oof_rows = [
            {
                "text": train_one,
                "gold_spans": [_span(train_one, "Rua Alfa", "Location"), _span(train_one, "João", "Person")],
                "pred_spans_eval": [_span(train_one, "Rua Alfa", "Location")],
            },
            {
                "text": train_three,
                "gold_spans": [_span(train_three, "Bairro Beta", "Location")],
                "pred_spans_eval": [],
            },
            {
                "text": false_positive_text,
                "gold_spans": [],
                "pred_spans_eval": [_span(false_positive_text, "Rua Gama", "Location")],
            },
        ]

        bucket_rows, combinations, summary = build_profile(
            train_rows,
            oof_rows,
            target_label="Location",
            pred_field="pred_spans_eval",
            min_bucket_support=1,
        )

        overall = next(row for row in bucket_rows if row["bucket_type"] == "all")
        self.assertEqual(overall["gold_support"], 2)
        self.assertEqual(overall["pred_support"], 2)
        self.assertEqual(overall["exact_tp"], 1)
        self.assertEqual(overall["false_negative"], 1)
        self.assertEqual(overall["false_positive"], 1)
        self.assertEqual(overall["f1"], 0.5)

        frequency_two = next(
            row
            for row in bucket_rows
            if row["bucket_type"] == "train_mention_frequency" and row["bucket"] == "2"
        )
        self.assertEqual(frequency_two["gold_support"], 1)
        self.assertEqual(frequency_two["exact_tp"], 1)

        designation_bairro = next(
            row for row in bucket_rows if row["bucket_type"] == "designator" and row["bucket"] == "bairro"
        )
        self.assertEqual(designation_bairro["false_negative"], 1)

        combination = next(row for row in combinations if row["label_combination"] == "Location + Person")
        self.assertEqual(combination["reports"], 1)
        self.assertEqual(summary["train"]["target_unique_mentions"], 2)
        self.assertEqual(summary["train"]["target_singleton_mentions"], 1)


if __name__ == "__main__":
    unittest.main()
