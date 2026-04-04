import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault("torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False), device=lambda x: x))
sys.modules.setdefault("base_model_training.cli", types.SimpleNamespace(parse_thresholds=lambda raw: [float(x) for x in str(raw).split(",") if x]))
sys.modules.setdefault(
    "base_model_training.cv",
    types.SimpleNamespace(
        _build_refit_split=None,
        _build_seen_entity_keys=None,
        _compute_seen_unseen_breakdown=None,
        _evaluate_thresholds=None,
        _load_model=None,
        _prepare_char_offsets=None,
        _run_single_training=None,
        clear_cuda_cache=lambda: None,
        materialize_model_base=lambda x: x,
        set_seed=lambda x: None,
    ),
)
sys.modules.setdefault(
    "base_model_training.data",
    types.SimpleNamespace(load_dataset=None, split_long_sentences=None),
)
sys.modules.setdefault("base_model_training.evaluate", types.SimpleNamespace(predict_entities_for_text=None))
sys.modules.setdefault("base_model_training.io_utils", types.SimpleNamespace(save_jsonl=lambda *args, **kwargs: None))
sys.modules.setdefault("base_model_training.paths", types.SimpleNamespace(resolve_path=lambda *_args: None))
sys.modules.setdefault(
    "pseudolabelling.evaluate_refit_pipeline",
    types.SimpleNamespace(
        compute_span_metrics=None,
        format_classification_report=None,
        load_gt_jsonl_strict=None,
    ),
)
sys.modules.setdefault("tools.inspect_dense_tips", types.SimpleNamespace(read_json_or_jsonl=None))

from base_model_training.train_quick import _convert_entity_rows_to_spans, _merge_training_rows


class BaseModelTrainQuickTests(unittest.TestCase):
    def test_convert_entity_rows_to_spans_converts_adjudicated_entities(self):
        rows = [
            {
                "text": "Ivete Sangalo em Salvador",
                "entities": [
                    {"text": "Ivete Sangalo", "label": "Person", "start": 0, "end": 13},
                    {"text": "Salvador", "label": "Location", "start": 17, "end": 25},
                ],
            }
        ]
        converted = _convert_entity_rows_to_spans(rows)
        self.assertEqual(
            converted,
            [
                {
                    "text": "Ivete Sangalo em Salvador",
                    "spans": [
                        {"start": 0, "end": 13, "label": "Person"},
                        {"start": 17, "end": 25, "label": "Location"},
                    ],
                }
            ],
        )

    def test_merge_training_rows_prefers_supervised_on_duplicate_text(self):
        supervised = [{"text": "A", "spans": [{"start": 0, "end": 1, "label": "Person"}]}]
        pseudolabels = [
            {"text": "A", "spans": [{"start": 0, "end": 1, "label": "Location"}]},
            {"text": "B", "spans": [{"start": 0, "end": 1, "label": "Organization"}]},
        ]
        merged = _merge_training_rows(
            supervised,
            pseudolabels,
            train_mode="supervised_plus_pseudolabels",
            deduplicate_by_text=True,
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["spans"][0]["label"], "Person")
        self.assertEqual(merged[1]["text"], "B")


if __name__ == "__main__":
    unittest.main()
