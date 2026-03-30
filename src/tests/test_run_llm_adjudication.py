import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_llm_adjudication import (
    ADJUDICATION_SCHEMA,
    build_messages,
    build_request_body,
    parse_adjudication_response,
)


class TestRunLlmAdjudication(unittest.TestCase):
    def test_build_messages_includes_text_and_candidate_data(self):
        row = {
            "source_id": "tip-1",
            "text": "Ivete Sangalo mora em Salvador",
            "baseline_entities": [{"text": "Ivete Sangalo", "label": "Person", "ner_score": 0.99}],
            "gliner2_entities": [{"text": "Ivete Sangalo", "label": "Person"}],
            "agreed_entities": [{"text": "Ivete Sangalo", "label": "Person", "match_type": "exact", "consensus_score": 1.0}],
            "baseline_only_entities": [],
            "gliner2_only_entities": [],
            "conflicts": [],
            "review_seed_entities": [{"text": "Ivete Sangalo", "label": "Person", "seed_origin": "agreed_exact"}],
            "metadata": {"agreement_ratio": 1.0},
        }
        messages = build_messages(row)
        self.assertEqual(len(messages), 2)
        self.assertIn("Return JSON only", messages[0]["content"])
        self.assertIn("Ivete Sangalo mora em Salvador", messages[1]["content"])
        self.assertIn("baseline_entities", messages[1]["content"])

    def test_build_request_body_uses_json_schema(self):
        body = build_request_body({"text": "abc"}, model="gpt-5")
        self.assertEqual(body["model"], "gpt-5")
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertEqual(body["text"]["format"]["schema"], ADJUDICATION_SCHEMA)

    def test_parse_adjudication_response_reads_output_text(self):
        payload = {
            "output_text": '{"decision":"accept","review_confidence":"high","entities_final":[{"text":"Ivete Sangalo","label":"Person"}],"justification":"Consistent."}'
        }
        parsed = parse_adjudication_response(payload)
        self.assertEqual(parsed["decision"], "accept")
        self.assertEqual(parsed["entities_final"][0]["label"], "Person")

    def test_parse_adjudication_response_reads_output_array(self):
        payload = {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"decision":"reject","review_confidence":"low","entities_final":[],"justification":"Too noisy."}'
                        }
                    ]
                }
            ]
        }
        parsed = parse_adjudication_response(payload)
        self.assertEqual(parsed["decision"], "reject")


if __name__ == "__main__":
    unittest.main()
