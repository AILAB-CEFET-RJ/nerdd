import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text_chunking import effective_chunk_budget, split_text_fast


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


if __name__ == "__main__":
    unittest.main()
