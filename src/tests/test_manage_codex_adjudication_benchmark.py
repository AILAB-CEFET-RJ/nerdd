import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.manage_codex_adjudication_benchmark import _init_state


class TestManageCodexAdjudicationBenchmark(unittest.TestCase):
    def test_init_excludes_already_adjudicated_source_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.jsonl"
            history_path = temp_path / "history.jsonl"
            benchmark_dir = temp_path / "benchmark"

            input_rows = [
                {"source_id": "a", "text": "Rua A", "review_seed_entities": []},
                {"source_id": "b", "text": "Rua B", "review_seed_entities": []},
                {"source_id": "c", "text": "Rua C", "review_seed_entities": []},
            ]
            history_rows = [
                {"source_id": "b", "adjudication": {"decision": "accept", "entities_final": []}},
                {"source_id": "x", "adjudication": {"decision": "reject", "entities_final": []}},
            ]

            with input_path.open("w", encoding="utf-8") as handle:
                for row in input_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            with history_path.open("w", encoding="utf-8") as handle:
                for row in history_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            args = Namespace(
                input=str(input_path),
                benchmark_dir=str(benchmark_dir),
                benchmark_name="bench",
                chunk_size=2,
                annotation_mode="train_annotation_open",
                exclude_adjudicated_from=[str(history_path)],
            )

            _init_state(args)

            state = json.loads((benchmark_dir / "state.json").read_text(encoding="utf-8"))
            benchmark_rows = [
                json.loads(line)
                for line in (benchmark_dir / "benchmark_input.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(state["records_total"], 2)
            self.assertEqual(state["excluded_source_ids_count"], 2)
            self.assertEqual([row["source_id"] for row in benchmark_rows], ["a", "c"])


if __name__ == "__main__":
    unittest.main()
