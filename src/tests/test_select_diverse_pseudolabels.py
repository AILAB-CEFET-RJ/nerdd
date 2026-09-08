import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.select_diverse_pseudolabels import select_diverse_rows


def _row(text, locs, score):
    spans = []
    cursor = 0
    for loc in locs:
        start = text.find(loc, cursor)
        if start < 0:
            start = len(text)
            text = f"{text} {loc}"
        end = start + len(loc)
        spans.append({"start": start, "end": end, "label": "Location", "text": loc})
        cursor = end
    return {
        "text": text,
        "spans": spans,
        "_pseudolabel": {"record_score_location": score},
    }


class SelectDiversePseudolabelsTests(unittest.TestCase):
    def test_selects_rows_with_score_inside_pseudolabel_metadata(self):
        rows = [_row("trafico em Mesquita", ["Mesquita"], 0.9)]
        selected, _audit, summary = select_diverse_rows(
            rows,
            top_n=1,
            score_fields=["record_score_location"],
            min_score=0.8,
            text_fields=["text"],
            span_keys=["spans"],
            label_field="label",
            target_labels={"Location"},
            max_per_entity=10,
            max_per_signature=2,
            signature_max_terms=8,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(summary["rows_selected"], 1)
        self.assertEqual(selected[0]["_pseudolabel_selection"]["score_field"], "_pseudolabel.record_score_location")

    def test_selects_rows_with_explicit_nested_score_field(self):
        rows = [_row("trafico em Mesquita", ["Mesquita"], 0.9)]
        selected, _audit, summary = select_diverse_rows(
            rows,
            top_n=1,
            score_fields=["_pseudolabel.record_score_location"],
            min_score=0.8,
            text_fields=["text"],
            span_keys=["spans"],
            label_field="label",
            target_labels={"Location"},
            max_per_entity=10,
            max_per_signature=2,
            signature_max_terms=8,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(summary["rows_selected"], 1)
        self.assertEqual(selected[0]["_pseudolabel_selection"]["score_field"], "_pseudolabel.record_score_location")

    def test_entity_cap_limits_repeated_location_term(self):
        rows = [
            _row("um em Mesquita", ["Mesquita"], 0.9),
            _row("dois em Mesquita", ["Mesquita"], 0.9),
            _row("tres em Nilopolis", ["Nilopolis"], 0.9),
        ]
        selected, _audit, summary = select_diverse_rows(
            rows,
            top_n=3,
            score_fields=["record_score_location"],
            min_score=0.8,
            text_fields=["text"],
            span_keys=["spans"],
            label_field="label",
            target_labels={"Location"},
            max_per_entity=1,
            max_per_signature=0,
            signature_max_terms=8,
        )
        selected_texts = [row["text"] for row in selected]
        self.assertEqual(selected_texts, ["um em Mesquita", "tres em Nilopolis"])
        self.assertEqual(summary["counters"]["rejected_entity_cap"], 1)

    def test_signature_cap_limits_same_location_set(self):
        rows = [
            _row("um em Mesquita e Chatuba", ["Mesquita", "Chatuba"], 0.9),
            _row("dois em Mesquita e Chatuba", ["Mesquita", "Chatuba"], 0.9),
            _row("tres em Mesquita e Chatuba", ["Mesquita", "Chatuba"], 0.9),
        ]
        selected, _audit, summary = select_diverse_rows(
            rows,
            top_n=3,
            score_fields=["record_score_location"],
            min_score=0.8,
            text_fields=["text"],
            span_keys=["spans"],
            label_field="label",
            target_labels={"Location"},
            max_per_entity=0,
            max_per_signature=2,
            signature_max_terms=8,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(summary["counters"]["rejected_signature_cap"], 1)

    def test_duplicate_text_is_removed(self):
        rows = [
            _row("trafico em Mesquita", ["Mesquita"], 0.9),
            _row("Tráfico   em   mesquita", ["mesquita"], 0.9),
        ]
        selected, _audit, summary = select_diverse_rows(
            rows,
            top_n=2,
            score_fields=["record_score_location"],
            min_score=0.8,
            text_fields=["text"],
            span_keys=["spans"],
            label_field="label",
            target_labels={"Location"},
            max_per_entity=0,
            max_per_signature=0,
            signature_max_terms=8,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(summary["counters"]["rejected_duplicate_text"], 1)


if __name__ == "__main__":
    unittest.main()
