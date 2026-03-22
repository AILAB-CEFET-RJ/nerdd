import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pseudolabelling.refit_pipeline import (
    _resolve_refit_sources,
    build_refit_splits,
    extract_text,
    merge_training_sources,
    normalize_entities,
    prepare_training_records,
    resolve_pseudolabel_input,
    split_train_val,
)


class RefitPipelineTests(unittest.TestCase):
    def test_extract_text_priority(self):
        record = {"relato": "abc", "text": "xyz"}
        text, source = extract_text(record, text_keys=("text", "relato"))
        self.assertEqual(text, "xyz")
        self.assertEqual(source, "text")

    def test_normalize_entities_filters_invalid(self):
        record = {
            "entities": [
                {"start": 0, "end": 3, "label": "Person"},
                {"start": "x", "end": 4, "label": "Person"},
                {"start": 2, "end": 1, "label": "Person"},
                {"start": 0, "end": 4, "label": ""},
                {"start": 0, "end": 4, "label": "Misc"},
            ]
        }
        entities, source, counters = normalize_entities(record, allowed_labels={"Person"})
        self.assertEqual(source, "entities")
        self.assertEqual(len(entities), 1)
        self.assertEqual(counters["invalid_span"], 2)
        self.assertEqual(counters["invalid_label"], 1)
        self.assertEqual(counters["filtered_by_label"], 1)

    def test_prepare_training_records(self):
        rows = [
            {"relato": "abc", "entities": [{"start": 0, "end": 3, "label": "Person"}]},
            {"relato": "", "entities": [{"start": 0, "end": 1, "label": "Person"}]},
            {"relato": "x", "entities": []},
        ]
        prepared, counters = prepare_training_records(rows, allowed_labels={"Person"})
        self.assertEqual(len(prepared), 1)
        self.assertEqual(counters["input_records"], 3)
        self.assertEqual(counters["missing_text"], 1)
        self.assertEqual(counters["dropped_no_entities"], 1)
        self.assertEqual(counters["kept_records"], 1)
        self.assertEqual(prepared[0]["_refit_input_meta"]["training_source"], "unknown")

    def test_prepare_training_records_tracks_source(self):
        rows = [{"relato": "abc", "entities": [{"start": 0, "end": 3, "label": "Person"}]}]
        prepared, _counters = prepare_training_records(rows, allowed_labels={"Person"}, source_name="supervised")
        self.assertEqual(prepared[0]["_refit_input_meta"]["training_source"], "supervised")

    def test_split_train_val(self):
        records = [{"id": i} for i in range(10)]
        train, val = split_train_val(records, val_ratio=0.2, seed=42)
        self.assertEqual(len(train) + len(val), 10)
        self.assertGreaterEqual(len(val), 1)
        train2, val2 = split_train_val(records, val_ratio=0.2, seed=42)
        self.assertEqual(train, train2)
        self.assertEqual(val, val2)

    def test_merge_training_sources_prefers_supervised_on_duplicate_text(self):
        supervised = [
            {
                "text": "same report",
                "entities": [{"start": 0, "end": 4, "label": "Person"}],
                "_refit_input_meta": {"training_source": "supervised"},
            }
        ]
        pseudolabel = [
            {
                "text": "same report",
                "entities": [{"start": 0, "end": 4, "label": "Person"}],
                "_refit_input_meta": {"training_source": "pseudolabel"},
            },
            {
                "text": "new report",
                "entities": [{"start": 0, "end": 3, "label": "Location"}],
                "_refit_input_meta": {"training_source": "pseudolabel"},
            },
        ]

        merged, counters = merge_training_sources(supervised, pseudolabel, deduplicate_by_text=True)
        self.assertEqual(len(merged), 2)
        self.assertEqual(counters["kept_supervised"], 1)
        self.assertEqual(counters["kept_pseudolabel"], 1)
        self.assertEqual(counters["dropped_duplicate_pseudolabel"], 1)
        self.assertEqual(merged[0]["_refit_input_meta"]["training_source"], "supervised")

    def test_merge_training_sources_can_skip_deduplication(self):
        supervised = [{"text": "same report", "_refit_input_meta": {"training_source": "supervised"}}]
        pseudolabel = [{"text": "same report", "_refit_input_meta": {"training_source": "pseudolabel"}}]
        merged, counters = merge_training_sources(supervised, pseudolabel, deduplicate_by_text=False)
        self.assertEqual(len(merged), 2)
        self.assertEqual(counters["kept_supervised"], 1)
        self.assertEqual(counters["kept_pseudolabel"], 1)

    def test_resolve_refit_sources(self):
        class Dummy:
            refit_mode = "supervised_only"

        self.assertEqual(_resolve_refit_sources(Dummy()), (True, False))
        Dummy.refit_mode = "supervised_plus_pseudolabels"
        self.assertEqual(_resolve_refit_sources(Dummy()), (True, True))
        Dummy.refit_mode = "pseudolabel_only"
        self.assertEqual(_resolve_refit_sources(Dummy()), (False, True))

    def test_resolve_pseudolabel_input_prefers_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            explicit = root / "accumulated.jsonl"
            explicit.write_text('{"text":"x"}\n', encoding="utf-8")
            split_dir = root / "split"
            split_dir.mkdir()
            (split_dir / "kept.jsonl").write_text('{"text":"y"}\n', encoding="utf-8")

            class Dummy:
                pseudolabel_path = str(explicit)
                input_path = str(split_dir)

            resolved = resolve_pseudolabel_input(root, Dummy())
            self.assertEqual(resolved, explicit)

    def test_build_refit_splits_keeps_validation_supervised_only(self):
        supervised_rows = [
            {"text": f"sup-{i}", "_refit_input_meta": {"training_source": "supervised"}}
            for i in range(10)
        ]
        pseudolabel_rows = [
            {"text": f"pseudo-{i}", "_refit_input_meta": {"training_source": "pseudolabel"}}
            for i in range(3)
        ]

        train_rows, val_rows, _merge_counts, _val_counts = build_refit_splits(
            supervised_rows,
            pseudolabel_rows,
            include_supervised_train=True,
            include_pseudolabel_train=True,
            val_ratio=0.2,
            seed=42,
            deduplicate_by_text=True,
        )

        self.assertTrue(train_rows)
        self.assertTrue(val_rows)
        self.assertTrue(all(row["_refit_input_meta"]["training_source"] == "supervised" for row in val_rows))
        self.assertEqual(
            sum(1 for row in train_rows if row["_refit_input_meta"]["training_source"] == "pseudolabel"),
            3,
        )

    def test_build_refit_splits_preserves_supervised_split_across_modes(self):
        supervised_rows = [
            {"text": f"sup-{i}", "_refit_input_meta": {"training_source": "supervised"}}
            for i in range(10)
        ]
        pseudolabel_rows = [
            {"text": f"pseudo-{i}", "_refit_input_meta": {"training_source": "pseudolabel"}}
            for i in range(3)
        ]

        train_sup_only, val_sup_only, _merge_a, _val_a = build_refit_splits(
            supervised_rows,
            pseudolabel_rows,
            include_supervised_train=True,
            include_pseudolabel_train=False,
            val_ratio=0.2,
            seed=42,
            deduplicate_by_text=True,
        )
        train_sup_plus, val_sup_plus, _merge_b, _val_b = build_refit_splits(
            supervised_rows,
            pseudolabel_rows,
            include_supervised_train=True,
            include_pseudolabel_train=True,
            val_ratio=0.2,
            seed=42,
            deduplicate_by_text=True,
        )

        sup_only_train_texts = {
            row["text"] for row in train_sup_only if row["_refit_input_meta"]["training_source"] == "supervised"
        }
        sup_plus_train_texts = {
            row["text"] for row in train_sup_plus if row["_refit_input_meta"]["training_source"] == "supervised"
        }
        self.assertEqual(sup_only_train_texts, sup_plus_train_texts)
        self.assertEqual({row["text"] for row in val_sup_only}, {row["text"] for row in val_sup_plus})


if __name__ == "__main__":
    unittest.main()
