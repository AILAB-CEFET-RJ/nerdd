import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_annotation_agreement_sample import (
    build_editor_records,
    filter_candidates,
    sample_candidates,
)
from tools.build_ner_annotation_editor_global import normalize_records


class BuildAnnotationAgreementSampleTests(unittest.TestCase):
    def test_builds_editor_compatible_unannotated_records(self):
        rows = [
            {"relato": "Traficantes na Rua Alpha", "spans": [{"start": 15, "end": 24, "label": "Location"}]},
            {"relato": "Fulano atua na comunidade Beta"},
            {"relato": "Policia recebeu denuncia"},
        ]
        candidates = filter_candidates(rows, text_field="auto", min_chars=1, max_chars=0)
        sampled = sample_candidates(candidates, sample_size=2, seed=3, preserve_input_order=True)
        records = build_editor_records(
            sampled,
            input_path="input.jsonl",
            seed=3,
            keep_existing_spans=False,
        )

        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["spans"] == [] for record in records))
        self.assertTrue(all(record["text"] for record in records))
        self.assertIn("_agreement_sample", records[0])

        normalized = normalize_records(records)
        self.assertEqual(normalized[0]["text"], records[0]["text"])
        self.assertEqual(normalized[0]["spans"], [])

    def test_can_keep_existing_spans_when_requested(self):
        rows = [{"id": "a", "text": "Rua Alpha", "entities": [{"start": 0, "end": 9, "label": "Location"}]}]
        candidates = filter_candidates(rows, text_field="auto", min_chars=1, max_chars=0)
        records = build_editor_records(
            candidates,
            input_path="input.json",
            seed=42,
            keep_existing_spans=True,
        )

        self.assertEqual(records[0]["spans"], [{"start": 0, "end": 9, "label": "Location"}])
        self.assertEqual(records[0]["_agreement_sample"]["source_ref"], "a")


if __name__ == "__main__":
    unittest.main()
