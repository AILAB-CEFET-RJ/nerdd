import sys
import unittest
from pathlib import Path
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault("torch", types.SimpleNamespace(tensor=lambda *args, **kwargs: None, double=None))
sys.modules.setdefault(
    "torch.utils.data",
    types.SimpleNamespace(DataLoader=object, WeightedRandomSampler=object),
)

from base_model_training.data import process_sample, split_long_sentences, tokenize_with_spans
from base_model_training.data import _build_non_empty_batches


class BaseModelTrainingDataTests(unittest.TestCase):
    def test_regex_tokenization_separates_punctuation(self):
        tokens, spans = tokenize_with_spans("Ivete Sangalo, Xuxa.", strategy="regex")

        self.assertEqual(tokens, ["Ivete", "Sangalo", ",", "Xuxa", "."])
        self.assertEqual(spans[1], (6, 13))
        self.assertEqual(spans[2], (13, 14))

    def test_process_sample_regex_keeps_entity_boundary_without_trailing_comma(self):
        sample = {
            "text": "Ivete Sangalo, Xuxa",
            "spans": [{"start": 0, "end": 13, "label": "Person"}],
            "sample_id": "s1",
        }

        processed = process_sample(sample, tokenization_strategy="regex")

        self.assertEqual(processed["tokenized_text"], ["Ivete", "Sangalo", ",", "Xuxa"])
        self.assertEqual(processed["ner"], [[0, 1, "Person"]])

    def test_split_long_sentences_can_keep_empty_chunks(self):
        dataset = [
            {
                "tokenized_text": ["a", "b", "c", "d"],
                "ner": [[0, 0, "Person"]],
                "sample_id": "s1",
            }
        ]

        chunks = split_long_sentences(
            dataset,
            max_length=2,
            overlap=0,
            tokenizer=None,
            keep_empty_chunks=True,
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["ner"], [[0, 0, "Person"]])
        self.assertEqual(chunks[1]["ner"], [])

    def test_split_long_sentences_drops_empty_chunks_by_default(self):
        dataset = [
            {
                "tokenized_text": ["a", "b", "c", "d"],
                "ner": [[0, 0, "Person"]],
                "sample_id": "s1",
            }
        ]

        chunks = split_long_sentences(
            dataset,
            max_length=2,
            overlap=0,
            tokenizer=None,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["ner"], [[0, 0, "Person"]])

    def test_split_long_sentences_respects_raw_token_cap(self):
        class _Tokenizer:
            def num_special_tokens_to_add(self, pair=False):
                return 2

            def __call__(self, words, is_split_into_words=True, add_special_tokens=False, truncation=False, return_attention_mask=False):
                del is_split_into_words, add_special_tokens, truncation, return_attention_mask

                class _Encoded:
                    def word_ids(self_nonlocal):
                        return []

                return _Encoded()

            def tokenize(self, word):
                del word
                return []

        dataset = [
            {
                "tokenized_text": [f"w{i}" for i in range(500)],
                "ner": [[0, 10, "Person"]],
                "sample_id": "s1",
            }
        ]

        chunks = split_long_sentences(
            dataset,
            max_length=384,
            overlap=0,
            tokenizer=_Tokenizer(),
        )

        self.assertEqual(len(chunks), 1)
        self.assertLessEqual(len(chunks[0]["tokenized_text"]), 384)

    def test_non_empty_batches_never_create_all_negative_batch(self):
        rng = __import__("random").Random(42)
        dataset = [
            {"ner": [[0, 0, "Person"]]},
            {"ner": [[0, 0, "Location"]]},
            {"ner": []},
            {"ner": []},
            {"ner": []},
        ]

        batches = _build_non_empty_batches(dataset, batch_size=2, rng=rng)

        self.assertTrue(batches)
        for batch in batches:
            self.assertTrue(any(dataset[index]["ner"] for index in batch))

    def test_non_empty_batches_cap_negatives_when_needed(self):
        rng = __import__("random").Random(1)
        dataset = [{"ner": [[0, 0, "Person"]]}] + [{"ner": []} for _ in range(10)]

        batches = _build_non_empty_batches(dataset, batch_size=2, rng=rng)

        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 2)
        self.assertTrue(any(dataset[index]["ner"] for index in batches[0]))


if __name__ == "__main__":
    unittest.main()
