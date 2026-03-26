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


if __name__ == "__main__":
    unittest.main()
