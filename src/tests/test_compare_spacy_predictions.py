import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_spacy_predictions import map_spacy_entities, parse_label_map


class DummyEnt:
    def __init__(self, start_char, end_char, label_, text):
        self.start_char = start_char
        self.end_char = end_char
        self.label_ = label_
        self.text = text


class DummyDoc:
    def __init__(self, ents):
        self.ents = ents


class CompareSpacyPredictionsTests(unittest.TestCase):
    def test_default_label_map_covers_person_org_gpe(self):
        mapping = parse_label_map("")
        self.assertEqual(mapping["PERSON"], "Person")
        self.assertEqual(mapping["ORG"], "Organization")
        self.assertEqual(mapping["GPE"], "Location")

    def test_map_spacy_entities_filters_unknown_labels(self):
        doc = DummyDoc(
            [
                DummyEnt(0, 4, "PERSON", "Joao"),
                DummyEnt(10, 19, "ORG", "Empresa X"),
                DummyEnt(25, 34, "DATE", "ontem"),
            ]
        )
        entities = map_spacy_entities(doc, parse_label_map(""))
        self.assertEqual(
            entities,
            [
                {"start": 0, "end": 4, "label": "Person", "text": "Joao"},
                {"start": 10, "end": 19, "label": "Organization", "text": "Empresa X"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
