import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault("gliner", types.SimpleNamespace(GLiNER=object))
sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda *args, **kwargs: args[0] if args else None))
sys.modules.setdefault(
    "calibration.serialization",
    types.SimpleNamespace(
        apply_calibrator_to_score=lambda score, label, calibrator: score,
        load_calibrator=lambda path: {},
    ),
)

from text_chunking import effective_chunk_budget, split_text_fast
from pseudolabelling.pipeline import predict_entities_for_texts


class _FakeTokenizer:
    model_max_length = 512

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(text.split())))

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(f"w{token_id + 1}" for token_id in token_ids)


class PredictionChunkingTests(unittest.TestCase):
    def test_effective_budget_respects_model_processor_max_len(self):
        tokenizer = _FakeTokenizer()
        model = SimpleNamespace(data_processor=SimpleNamespace(transformer_tokenizer=tokenizer, max_len=4))
        self.assertEqual(effective_chunk_budget(model, tokenizer, 10), 2)

    def test_split_text_fast_uses_effective_budget(self):
        tokenizer = _FakeTokenizer()
        model = SimpleNamespace(data_processor=SimpleNamespace(transformer_tokenizer=tokenizer, max_len=4))
        text = "w1 w2 w3 w4 w5 w6"

        chunks = split_text_fast(text, model=model, tokenizer=tokenizer, max_tokens=10)

        self.assertEqual(chunks, ["w1 w2", "w3 w4", "w5 w6"])

    def test_split_text_fast_uses_requested_limit_when_lower(self):
        tokenizer = _FakeTokenizer()
        model = SimpleNamespace(data_processor=SimpleNamespace(transformer_tokenizer=tokenizer, max_len=10))
        text = "w1 w2 w3 w4"

        chunks = split_text_fast(text, model=model, tokenizer=tokenizer, max_tokens=3)

        self.assertEqual(chunks, ["w1", "w2", "w3", "w4"])

    def test_predict_entities_for_texts_batches_across_rows(self):
        tokenizer = _FakeTokenizer()
        model = SimpleNamespace(data_processor=SimpleNamespace(transformer_tokenizer=tokenizer, max_len=10))
        calls = []

        def fake_predict_batch_entities(_model, batch_texts, labels, threshold):
            del _model, labels, threshold
            calls.append(list(batch_texts))
            return [
                [{"start": 0, "end": 2, "label": "Person", "text": "w1", "score": 0.9}]
                for _ in batch_texts
            ]

        with patch("pseudolabelling.pipeline.predict_batch_entities", side_effect=fake_predict_batch_entities):
            entities_by_row = predict_entities_for_texts(
                model=model,
                texts=["w1 w2", "w1 w2"],
                labels=["Person"],
                batch_size=2,
                max_tokens=10,
                score_threshold=0.0,
            )

        self.assertEqual(calls, [["w1 w2", "w1 w2"]])
        self.assertEqual(len(entities_by_row), 2)
        self.assertEqual(entities_by_row[0][0]["label"], "Person")
        self.assertEqual(entities_by_row[1][0]["label"], "Person")


if __name__ == "__main__":
    unittest.main()
