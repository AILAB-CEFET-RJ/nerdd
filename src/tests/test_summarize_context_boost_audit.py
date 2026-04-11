import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestSummarizeContextBoostAudit(unittest.TestCase):
    def test_summarize_boosted_entities_jsonl_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "boosted.jsonl"
            summary_path = temp / "summary.json"
            rows_csv = temp / "rows.csv"

            input_rows = [
                {
                    "row_index": 1,
                    "sample_id": "tip-1",
                    "entity_text": "Rua Faia",
                    "label": "Location",
                    "score_before": 0.8,
                    "score_after": 0.96,
                    "boost_multiplier": 1.2,
                    "boost_reason": "location-entity-overlaps-metadata",
                    "matched_metadata_fields": ["logradouroLocal"],
                    "matched_metadata_values": ["Rua Faia"],
                },
                {
                    "row_index": 2,
                    "sample_id": "tip-2",
                    "entity_text": "Centro",
                    "label": "Location",
                    "score_before": 0.7,
                    "score_after": 0.84,
                    "boost_multiplier": 1.2,
                    "boost_reason": "location-entity-overlaps-metadata",
                    "matched_metadata_fields": ["bairroLocal"],
                    "matched_metadata_values": ["Centro"],
                },
            ]
            with input_path.open("w", encoding="utf-8") as handle:
                for row in input_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    "src/tools/summarize_context_boost_audit.py",
                    "--boosted-entities-jsonl",
                    str(input_path),
                    "--summary-json",
                    str(summary_path),
                    "--rows-csv",
                    str(rows_csv),
                    "--top-n",
                    "1",
                ],
                check=True,
                cwd="/home/ebezerra/ailab/nerdd",
            )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["input_mode"], "boosted_entities_jsonl")
            self.assertEqual(summary["boosted_entities_total"], 2)
            self.assertEqual(summary["boosted_label_counts"]["Location"], 2)
            self.assertAlmostEqual(summary["avg_score_delta"], 0.15, places=6)

            with rows_csv.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["entity_label"], "Location")


if __name__ == "__main__":
    unittest.main()
