import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.find_missing_repeated_spans import find_missing_repeated_spans


class FindMissingRepeatedSpansTests(unittest.TestCase):
    def test_finds_exact_unannotated_occurrence(self):
        rows = [
            {
                "text": "Rua A tem denúncia",
                "spans": [{"start": 0, "end": 5, "label": "Location"}],
            },
            {
                "text": "Passei pela Rua A ontem",
                "spans": [],
            },
        ]

        report = find_missing_repeated_spans(rows)

        self.assertEqual(report["missing_candidates_total"], 1)
        candidate = report["candidates"][0]
        self.assertEqual(candidate["row_index_0based"], 1)
        self.assertEqual(candidate["row_index_1based"], 2)
        self.assertEqual(candidate["mention"], "Rua A")
        self.assertEqual(candidate["label"], "Location")
        self.assertEqual(candidate["start"], 12)
        self.assertEqual(candidate["end"], 17)

    def test_ignores_occurrence_already_annotated_with_same_label(self):
        rows = [
            {
                "text": "Rua A e Rua A",
                "spans": [
                    {"start": 0, "end": 5, "label": "Location"},
                    {"start": 8, "end": 13, "label": "Location"},
                ],
            }
        ]

        report = find_missing_repeated_spans(rows)

        self.assertEqual(report["missing_candidates_total"], 0)
        self.assertEqual(report["conflicts_total"], 0)

    def test_ignores_occurrence_contained_in_larger_same_label_span(self):
        rows = [
            {
                "text": "Antônio João aparece em outro relato",
                "spans": [{"start": 0, "end": 12, "label": "Location"}],
            },
            {
                "text": "boca de fumo na rua Antônio João , favela da tinta",
                "spans": [{"start": 16, "end": 32, "label": "Location"}],
            },
        ]

        report = find_missing_repeated_spans(rows)

        self.assertEqual(report["missing_candidates_total"], 0)
        self.assertEqual(report["conflicts_total"], 0)

    def test_ignores_occurrence_overlapping_same_label_span(self):
        rows = [
            {
                "text": "São Gonçalo -RJ tem relato",
                "spans": [{"start": 12, "end": 15, "label": "Location"}],
            },
            {
                "text": "roubos em Nilópolis-RJ",
                "spans": [
                    {"start": 10, "end": 19, "label": "Location"},
                    {"start": 20, "end": 22, "label": "Location"},
                ],
            },
        ]

        report = find_missing_repeated_spans(rows)

        self.assertEqual(report["missing_candidates_total"], 0)
        self.assertEqual(report["conflicts_total"], 0)

    def test_does_not_match_inside_alphanumeric_sequence(self):
        rows = [
            {
                "text": "O 7° BPM atua aqui",
                "spans": [{"start": 2, "end": 8, "label": "Organization"}],
            },
            {
                "text": "O 27° BPM atua aqui",
                "spans": [],
            },
            {
                "text": "O 7° BPM voltou",
                "spans": [],
            },
        ]

        report = find_missing_repeated_spans(rows)

        candidates = [
            candidate
            for candidate in report["candidates"]
            if candidate["inventory_mention"] == "7° BPM"
        ]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["row_index_0based"], 2)
        self.assertEqual(candidates[0]["mention"], "7° BPM")

    def test_does_not_candidate_occurrence_inside_other_annotated_span(self):
        rows = [
            {
                "text": "Angra aparece como lugar",
                "spans": [{"start": 0, "end": 5, "label": "Location"}],
            },
            {
                "text": "Gap do Ministério público de Angra",
                "spans": [{"start": 0, "end": 33, "label": "Organization"}],
            },
        ]

        report = find_missing_repeated_spans(rows)

        self.assertEqual(report["missing_candidates_total"], 0)
        self.assertEqual(report["conflicts_total"], 0)

    def test_reports_conflict_when_offsets_have_another_label(self):
        rows = [
            {
                "text": "Ada",
                "spans": [{"start": 0, "end": 3, "label": "Person"}],
            },
            {
                "text": "Ada",
                "spans": [{"start": 0, "end": 3, "label": "Organization"}],
            },
        ]

        report = find_missing_repeated_spans(rows)

        self.assertEqual(report["missing_candidates_total"], 0)
        self.assertEqual(report["conflicts_total"], 2)
        expected_labels = {conflict["expected_label"] for conflict in report["conflicts"]}
        self.assertEqual(expected_labels, {"Person", "Organization"})

    def test_respects_min_len_and_label_filter(self):
        rows = [
            {
                "text": "PM viu Rua A",
                "spans": [
                    {"start": 0, "end": 2, "label": "Organization"},
                    {"start": 7, "end": 12, "label": "Location"},
                ],
            },
            {
                "text": "PM passou na Rua A",
                "spans": [],
            },
        ]

        report = find_missing_repeated_spans(rows, labels={"Location"}, min_len=3)

        self.assertEqual(report["missing_candidates_total"], 1)
        self.assertEqual(report["candidates"][0]["mention"], "Rua A")
        self.assertEqual(report["candidates"][0]["label"], "Location")

    def test_ignore_case_keeps_original_occurrence_text(self):
        rows = [
            {
                "text": "Rua A",
                "spans": [{"start": 0, "end": 5, "label": "Location"}],
            },
            {
                "text": "rua a",
                "spans": [],
            },
        ]

        report = find_missing_repeated_spans(rows, ignore_case=True)

        self.assertEqual(report["missing_candidates_total"], 1)
        self.assertEqual(report["candidates"][0]["mention"], "rua a")
        self.assertEqual(report["candidates"][0]["inventory_mention"], "Rua A")


if __name__ == "__main__":
    unittest.main()
