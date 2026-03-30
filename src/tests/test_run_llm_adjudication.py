import unittest
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_llm_adjudication import (
    ADJUDICATION_SCHEMA,
    AdjudicationValidationError,
    adjudicate_row,
    build_messages,
    build_request_body,
    load_dotenv,
    parse_adjudication_response,
    resolve_model_name,
    resolve_temperature,
    validate_adjudication,
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
        body = build_request_body({"text": "abc"}, model="gpt-5", temperature=0.7)
        self.assertEqual(body["model"], "gpt-5")
        self.assertEqual(body["temperature"], 0.7)
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

    def test_load_dotenv_reads_simple_key_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".env"
            path.write_text(
                "# comment\nOPENAI_API_KEY=test-key\nOPENAI_DEFAULT_MODEL=gpt-4o-mini\nOPENAI_DEFAULT_TEMPERATURE='0.7'\nOTHER=' spaced value '\n",
                encoding="utf-8",
            )
            values = load_dotenv(path)
            self.assertEqual(values["OPENAI_API_KEY"], "test-key")
            self.assertEqual(values["OTHER"], " spaced value ")
            self.assertEqual(resolve_model_name("", values), "gpt-4o-mini")
            self.assertEqual(resolve_temperature(None, values), 0.7)

    def test_validate_adjudication_requires_literal_substrings(self):
        source_row = {
            "text": "Belford Roxo rj",
            "review_seed_entities": [],
        }
        with self.assertRaises(ValueError):
            validate_adjudication(
                {
                    "decision": "accept_with_edits",
                    "review_confidence": "medium",
                    "entities_final": [{"text": "Belford Roxo RJ", "label": "Location"}],
                    "justification": "bad normalization",
                },
                source_row,
            )

    def test_validate_adjudication_accept_must_stay_within_review_seed_entities(self):
        source_row = {
            "text": "Ivete Sangalo em Salvador",
            "review_seed_entities": [{"text": "Ivete Sangalo", "label": "Person"}],
        }
        with self.assertRaises(ValueError):
            validate_adjudication(
                {
                    "decision": "accept",
                    "review_confidence": "high",
                    "entities_final": [
                        {"text": "Ivete Sangalo", "label": "Person"},
                        {"text": "Salvador", "label": "Location"},
                    ],
                    "justification": "too liberal",
                },
                source_row,
            )

    def test_validate_adjudication_accept_with_edits_must_stay_within_review_seed_entities(self):
        source_row = {
            "text": "Ivete Sangalo em Salvador",
            "review_seed_entities": [{"text": "Ivete Sangalo", "label": "Person"}],
        }
        with self.assertRaises(ValueError):
            validate_adjudication(
                {
                    "decision": "accept_with_edits",
                    "review_confidence": "medium",
                    "entities_final": [
                        {"text": "Ivete Sangalo", "label": "Person"},
                        {"text": "Salvador", "label": "Location"},
                    ],
                    "justification": "too liberal",
                },
                source_row,
            )

    def test_parse_adjudication_response_raises_validation_error_for_missing_output(self):
        with self.assertRaises(AdjudicationValidationError):
            parse_adjudication_response({})

    def test_adjudicate_row_downgrades_validation_failures_to_reject(self):
        from tools import run_llm_adjudication as module

        original_call = module.call_responses_api
        try:
            module.call_responses_api = lambda *args, **kwargs: {
                "output_text": (
                    '{"decision":"accept_with_edits","review_confidence":"medium",'
                    '"entities_final":[{"text":"Ivete Sangalo","label":"Person"},{"text":"Salvador","label":"Location"}],'
                    '"justification":"too liberal"}'
                )
            }
            result = adjudicate_row(
                {
                    "source_id": "tip-1",
                    "text": "Ivete Sangalo em Salvador",
                    "review_seed_entities": [{"text": "Ivete Sangalo", "label": "Person"}],
                },
                model="gpt-4o-mini",
                temperature=0.0,
                api_key="test-key",
                api_base="https://example.invalid/v1/responses",
                timeout_seconds=1,
                max_retries=3,
                retry_sleep_seconds=0.0,
            )
        finally:
            module.call_responses_api = original_call

        self.assertEqual(result["adjudication"]["decision"], "reject")
        self.assertEqual(result["adjudication"]["entities_final"], [])
        self.assertIn("Validation downgrade:", result["adjudication"]["justification"])


if __name__ == "__main__":
    unittest.main()
