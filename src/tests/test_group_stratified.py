import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.group_stratified import StratifiedGroupKFoldNER


class StratifiedGroupKFoldNERTests(unittest.TestCase):
    def test_preserves_groups_and_balances_labels(self):
        dataset = [
            {"sample_id": "g1", "ner": [[0, 0, "A"]], "tokenized_text": ["x"]},
            {"sample_id": "g2", "ner": [[0, 0, "A"]], "tokenized_text": ["x"]},
            {"sample_id": "g3", "ner": [[0, 0, "A"]], "tokenized_text": ["x"]},
            {"sample_id": "g4", "ner": [[0, 0, "B"]], "tokenized_text": ["x"]},
            {"sample_id": "g5", "ner": [[0, 0, "B"]], "tokenized_text": ["x"]},
            {"sample_id": "g6", "ner": [[0, 0, "B"]], "tokenized_text": ["x"]},
            {"sample_id": "g7", "ner": [[0, 0, "A"], [0, 0, "B"]], "tokenized_text": ["x"]},
            {"sample_id": "g8", "ner": [[0, 0, "A"], [0, 0, "B"]], "tokenized_text": ["x"]},
            {"sample_id": "g9", "ner": [[0, 0, "A"], [0, 0, "B"]], "tokenized_text": ["x"]},
        ]
        groups = [sample["sample_id"] for sample in dataset]

        splitter = StratifiedGroupKFoldNER(n_splits=3, seed=42)
        splits = list(splitter.split(dataset, groups=groups))

        self.assertEqual(len(splits), 3)

        seen_test_groups = set()
        for _train_idx, test_idx in splits:
            test_groups = {dataset[index]["sample_id"] for index in test_idx.tolist()}
            self.assertEqual(len(test_groups), len(test_idx))
            self.assertTrue(seen_test_groups.isdisjoint(test_groups))
            seen_test_groups.update(test_groups)

            labels = {label for index in test_idx.tolist() for _, _, label in dataset[index]["ner"]}
            self.assertIn("A", labels)
            self.assertIn("B", labels)

        summary = splitter.last_summary
        self.assertIsNotNone(summary)
        example_counts = [fold["example_count"] for fold in summary["folds"]]
        self.assertLessEqual(max(example_counts) - min(example_counts), 1)

    def test_never_emits_empty_fold(self):
        dataset = [
            {"sample_id": "g1", "ner": [[0, 0, "Location"]] * 10, "tokenized_text": ["x"]},
            {"sample_id": "g2", "ner": [[0, 0, "Location"]] * 8, "tokenized_text": ["x"]},
            {"sample_id": "g3", "ner": [[0, 0, "Location"]] * 6, "tokenized_text": ["x"]},
            {"sample_id": "g4", "ner": [[0, 0, "Organization"]] * 4, "tokenized_text": ["x"]},
            {"sample_id": "g5", "ner": [[0, 0, "Organization"]] * 2, "tokenized_text": ["x"]},
            {"sample_id": "g6", "ner": [[0, 0, "Person"]], "tokenized_text": ["x"]},
        ]
        groups = [sample["sample_id"] for sample in dataset]

        splitter = StratifiedGroupKFoldNER(n_splits=3, seed=42)
        splits = list(splitter.split(dataset, groups=groups))

        self.assertEqual(len(splits), 3)
        self.assertTrue(all(len(test_idx) > 0 for _train_idx, test_idx in splits))

        summary = splitter.last_summary
        self.assertIsNotNone(summary)
        self.assertTrue(all(fold["group_count"] > 0 for fold in summary["folds"]))
        self.assertTrue(all(fold["example_count"] > 0 for fold in summary["folds"]))
        group_counts = [fold["group_count"] for fold in summary["folds"]]
        self.assertLessEqual(max(group_counts) - min(group_counts), 1)

    def test_distributes_rare_label_groups_across_folds(self):
        dataset = [
            {"sample_id": "g1", "ner": [[0, 0, "A"]], "tokenized_text": ["x"]},
            {"sample_id": "g2", "ner": [[0, 0, "A"]], "tokenized_text": ["x"]},
            {"sample_id": "g3", "ner": [[0, 0, "A"]], "tokenized_text": ["x"]},
            {"sample_id": "g4", "ner": [[0, 0, "A"]], "tokenized_text": ["x"]},
            {"sample_id": "g5", "ner": [[0, 0, "A"]], "tokenized_text": ["x"]},
            {"sample_id": "g6", "ner": [[0, 0, "A"]], "tokenized_text": ["x"]},
            {"sample_id": "g7", "ner": [[0, 0, "B"]] * 5, "tokenized_text": ["x"]},
            {"sample_id": "g8", "ner": [[0, 0, "B"]] * 5, "tokenized_text": ["x"]},
            {"sample_id": "g9", "ner": [[0, 0, "B"]] * 5, "tokenized_text": ["x"]},
        ]
        groups = [sample["sample_id"] for sample in dataset]

        splitter = StratifiedGroupKFoldNER(n_splits=3, seed=42)
        list(splitter.split(dataset, groups=groups))

        summary = splitter.last_summary
        self.assertIsNotNone(summary)
        rare_label_spans = [fold["label_spans"]["B"] for fold in summary["folds"]]
        self.assertTrue(all(span_count >= 5 for span_count in rare_label_spans))


if __name__ == "__main__":
    unittest.main()
