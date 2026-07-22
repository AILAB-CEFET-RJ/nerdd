import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sample_unlabeled_by_target_assunto import (
    allocate_best_effort,
    load_pool_by_assunto,
    load_target_counts,
    sample_by_allocation,
    total_variation_distance,
    write_jsonl,
)


class SampleUnlabeledByTargetAssuntoTests(unittest.TestCase):
    def test_best_effort_exhausts_scarce_target_assunto_and_redistributes(self):
        target = Counter({"A": 50, "B": 30, "C": 20})
        pool = {"A": 100, "B": 100, "C": 2}

        allocations = allocate_best_effort(
            target_counts=target,
            pool_counts=pool,
            sample_size=20,
        )

        self.assertEqual(sum(allocations.values()), 20)
        self.assertEqual(allocations["C"], 2)
        self.assertGreater(allocations["A"], allocations["B"])

    def test_sampling_preserves_requested_allocation(self):
        pool = {
            "A": [(1, {"assunto": "A", "relato": "a1"}), (2, {"assunto": "A", "relato": "a2"})],
            "B": [(3, {"assunto": "B", "relato": "b1"})],
        }

        sampled = sample_by_allocation(
            pool,
            {"A": 1, "B": 1},
            seed=42,
            preserve_input_order=True,
        )

        self.assertEqual([line_no for line_no, _ in sampled], sorted(line_no for line_no, _ in sampled))
        self.assertEqual(Counter(row["assunto"] for _, row in sampled), {"A": 1, "B": 1})

    def test_loads_target_and_pool_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target_path = tmp_path / "target.json"
            pool_path = tmp_path / "pool.jsonl"
            target_path.write_text(
                json.dumps(
                    [
                        {"text": "x", "assunto": "A"},
                        {"text": "y", "assunto": "A"},
                        {"text": "z", "assunto": "B"},
                    ]
                ),
                encoding="utf-8",
            )
            write_jsonl(
                pool_path,
                [
                    {"relato": "a", "assunto": "A"},
                    {"relato": "b", "assunto": "B"},
                    {"relato": "c", "assunto": ""},
                ],
            )

            self.assertEqual(load_target_counts([str(target_path)], assunto_field="assunto"), {"A": 2, "B": 1})
            pool = load_pool_by_assunto(str(pool_path), assunto_field="assunto")
            self.assertEqual({key: len(value) for key, value in pool.items()}, {"A": 1, "B": 1})

    def test_total_variation_distance(self):
        self.assertAlmostEqual(total_variation_distance({"A": 1}, {"A": 1}), 0.0)
        self.assertAlmostEqual(total_variation_distance({"A": 1}, {"B": 1}), 1.0)


if __name__ == "__main__":
    unittest.main()
