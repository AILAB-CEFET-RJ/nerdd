import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_train_annotation_prompt_probe import main as probe_main


class BuildTrainAnnotationPromptProbeTests(unittest.TestCase):
    def test_builds_probe_rows_from_regressions_and_wins(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            regressions_path = root / "regressions.jsonl"
            wins_path = root / "wins.jsonl"
            source_path = root / "source.jsonl"
            output_path = root / "probe.jsonl"
            summary_path = root / "summary.json"

            regression_rows = [
                {
                    "text": "Rua Alpha com Joao",
                    "spans": [{"start": 0, "end": 9, "label": "Location", "text": "Rua Alpha"}],
                    "candidate_entities": [{"start": 0, "end": 9, "label": "Person", "text": "Rua Alpha"}],
                    "_audit": {"loss_reasons": {"wrong_label": 1}, "delta_row_f1": -1.0},
                },
                {
                    "text": "Rua Beta",
                    "spans": [{"start": 0, "end": 8, "label": "Location", "text": "Rua Beta"}],
                    "candidate_entities": [{"start": 4, "end": 8, "label": "Location", "text": "Beta"}],
                    "_audit": {"loss_reasons": {"boundary_or_partial": 1}, "delta_row_f1": -1.0},
                },
                {
                    "text": "Texto com espurio",
                    "spans": [],
                    "candidate_entities": [{"start": 0, "end": 5, "label": "Person", "text": "Texto"}],
                    "_audit": {"loss_reasons": {"spurious_entity": 1}, "delta_row_f1": -1.0},
                },
            ]
            wins_rows = [
                {
                    "text": "Boa win",
                    "_audit": {"win_reasons": {"recovered_exact_match": 1}, "delta_row_f1": 1.0},
                }
            ]
            source_rows = [
                {"source_id": "a", "text": "Rua Alpha com Joao", "review_seed_entities": [{"text": "Rua Alpha", "label": "Location"}]},
                {"source_id": "b", "text": "Rua Beta", "review_seed_entities": [{"text": "Rua Beta", "label": "Location"}]},
                {"source_id": "c", "text": "Texto com espurio", "review_seed_entities": []},
                {"source_id": "d", "text": "Boa win", "review_seed_entities": [{"text": "Boa", "label": "Organization"}]},
            ]
            regressions_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in regression_rows) + "\n", encoding="utf-8")
            wins_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in wins_rows) + "\n", encoding="utf-8")
            source_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in source_rows) + "\n", encoding="utf-8")

            old_argv = sys.argv[:]
            try:
                sys.argv = [
                    "build_train_annotation_prompt_probe.py",
                    "--regressions-jsonl",
                    str(regressions_path),
                    "--wins-jsonl",
                    str(wins_path),
                    "--source-input",
                    str(source_path),
                    "--output-jsonl",
                    str(output_path),
                    "--summary-json",
                    str(summary_path),
                    "--top-location-person",
                    "1",
                    "--top-location-org",
                    "0",
                    "--top-boundary",
                    "1",
                    "--top-spurious",
                    "1",
                    "--top-wins",
                    "1",
                ]
                probe_main()
            finally:
                sys.argv = old_argv

            probe_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(len(probe_rows), 4)
            self.assertEqual(summary["selected_rows"], 4)
            categories = set()
            for row in probe_rows:
                categories.update(row["_probe_meta"]["categories"])
            self.assertIn("location_to_person", categories)
            self.assertIn("boundary_or_partial", categories)
            self.assertIn("spurious_entity", categories)
            self.assertIn("win_reference", categories)

    def test_builds_probe_rows_without_source_input(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            regressions_path = root / "regressions.jsonl"
            wins_path = root / "wins.jsonl"
            output_path = root / "probe.jsonl"

            regression_rows = [
                {
                    "text": "Rua Alpha com Joao",
                    "spans": [{"start": 0, "end": 9, "label": "Location", "text": "Rua Alpha"}],
                    "candidate_entities": [{"start": 0, "end": 9, "label": "Person", "text": "Rua Alpha"}],
                    "_audit": {"loss_reasons": {"wrong_label": 1}, "delta_row_f1": -1.0},
                }
            ]
            wins_rows = [
                {
                    "text": "Boa win",
                    "_audit": {"win_reasons": {"recovered_exact_match": 1}, "delta_row_f1": 1.0},
                }
            ]
            regressions_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in regression_rows) + "\n", encoding="utf-8")
            wins_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in wins_rows) + "\n", encoding="utf-8")

            old_argv = sys.argv[:]
            try:
                sys.argv = [
                    "build_train_annotation_prompt_probe.py",
                    "--regressions-jsonl",
                    str(regressions_path),
                    "--wins-jsonl",
                    str(wins_path),
                    "--output-jsonl",
                    str(output_path),
                    "--top-location-person",
                    "1",
                    "--top-location-org",
                    "0",
                    "--top-boundary",
                    "0",
                    "--top-spurious",
                    "0",
                    "--top-wins",
                    "1",
                ]
                probe_main()
            finally:
                sys.argv = old_argv

            probe_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(probe_rows), 2)
            self.assertEqual(probe_rows[0]["text"], "Rua Alpha com Joao")
            self.assertEqual(probe_rows[0]["review_seed_entities"], [])


if __name__ == "__main__":
    unittest.main()
