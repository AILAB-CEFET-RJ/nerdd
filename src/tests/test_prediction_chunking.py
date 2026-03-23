import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text_chunking import split_text_encoder_aware


class _FakeBatchEncoding:
    def word_ids(self):
        raise RuntimeError("force tokenize() fallback")


class _FakeTokenizer:
    model_max_length = 512

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, *args, **kwargs):
        if kwargs.get("return_offsets_mapping"):
            text = args[0]
            offsets = []
            cursor = 0
            for token in text.split():
                start = text.find(token, cursor)
                end = start + len(token)
                offsets.append((start, end))
                cursor = end
            return {"offset_mapping": offsets}
        return _FakeBatchEncoding()

    def tokenize(self, word):
        return [word]


class PredictionChunkingTests(unittest.TestCase):
    def test_prediction_chunking_respects_model_processor_max_len(self):
        tokenizer = _FakeTokenizer()
        model = SimpleNamespace(data_processor=SimpleNamespace(transformer_tokenizer=tokenizer, max_len=4))
        text = "w1 w2 w3 w4 w5 w6"

        chunks = split_text_encoder_aware(text, model=model, tokenizer=tokenizer, max_tokens=10)

        self.assertEqual([chunk["text"] for chunk in chunks], ["w1 w2", "w3 w4", "w5 w6"])

    def test_chunking_uses_requested_limit_when_lower_than_model_limit(self):
        tokenizer = _FakeTokenizer()
        model = SimpleNamespace(data_processor=SimpleNamespace(transformer_tokenizer=tokenizer, max_len=10))
        text = "w1 w2 w3 w4"

        chunks = split_text_encoder_aware(text, model=model, tokenizer=tokenizer, max_tokens=3)

        self.assertEqual([chunk["text"] for chunk in chunks], ["w1", "w2", "w3", "w4"])


if __name__ == "__main__":
    unittest.main()
