import unittest
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_llm_adjudication import (
    ADJUDICATION_SCHEMA,
    AdjudicationValidationError,
    _load_existing_rows,
    _processed_source_ids,
    adjudicate_row,
    build_messages,
    build_request_body,
    load_dotenv,
    model_supports_temperature,
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

    def test_build_messages_train_annotation_allows_new_entities_instruction(self):
        row = {
            "source_id": "tip-1",
            "text": "Ivete Sangalo mora em Salvador",
            "review_seed_entities": [{"text": "Ivete Sangalo", "label": "Person", "start": 0, "end": 14}],
        }
        messages = build_messages(row, annotation_mode="train_annotation")
        self.assertIn("not restricted to review_seed_entities", messages[0]["content"])
        self.assertIn("locative marker is part of the literal mention", messages[0]["content"])
        self.assertIn("aligned with the corpus convention", messages[1]["content"])
        self.assertIn("Allow institutional mentions such as polícia as Organization", messages[1]["content"])

    def test_build_request_body_uses_json_schema(self):
        body = build_request_body({"text": "abc"}, model="gpt-4o-mini", temperature=0.7)
        self.assertEqual(body["model"], "gpt-4o-mini")
        self.assertEqual(body["temperature"], 0.7)
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertEqual(body["text"]["format"]["schema"], ADJUDICATION_SCHEMA)

    def test_build_request_body_propagates_annotation_mode(self):
        body = build_request_body(
            {"text": "Ivete Sangalo mora em Salvador"},
            model="gpt-4o-mini",
            temperature=0.0,
            annotation_mode="train_annotation",
        )
        self.assertIn("not restricted to review_seed_entities", body["input"][0]["content"])
        self.assertIn("Road and address markers", body["input"][0]["content"])

    def test_build_request_body_omits_temperature_for_gpt5_models(self):
        body = build_request_body({"text": "abc"}, model="gpt-5-mini", temperature=0.7)
        self.assertEqual(body["model"], "gpt-5-mini")
        self.assertNotIn("temperature", body)

    def test_model_supports_temperature_matches_gpt5_rules(self):
        self.assertFalse(model_supports_temperature("gpt-5"))
        self.assertFalse(model_supports_temperature("gpt-5-mini"))
        self.assertFalse(model_supports_temperature("gpt-5-nano"))
        self.assertTrue(model_supports_temperature("gpt-5.1"))
        self.assertTrue(model_supports_temperature("gpt-4o-mini"))

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

    def test_resume_helpers_load_existing_rows_and_processed_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "out.jsonl"
            output_path.write_text(
                (
                    '{"source_id":"tip-1","adjudication":{"decision":"reject"}}\n'
                    '{"source_id":"tip-2","adjudication":{"decision":"accept_with_edits"}}\n'
                ),
                encoding="utf-8",
            )
            error_rows = [{"source_id": "tip-3", "error": "boom"}]
            success_rows = _load_existing_rows(output_path)
            processed = _processed_source_ids(success_rows, error_rows)
            self.assertEqual(len(success_rows), 2)
            self.assertEqual(processed, {"tip-1", "tip-2", "tip-3"})

    def test_resolve_model_name_prefers_cli_over_dotenv(self):
        self.assertEqual(
            resolve_model_name("gpt-5-mini", {"OPENAI_DEFAULT_MODEL": "gpt-4o-mini"}),
            "gpt-5-mini",
        )

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

    def test_validate_adjudication_train_annotation_allows_new_literal_entities(self):
        source_row = {
            "text": "Ivete Sangalo em Salvador",
            "review_seed_entities": [{"text": "Ivete Sangalo", "label": "Person", "start": 0, "end": 13}],
        }
        validated = validate_adjudication(
            {
                "decision": "accept_with_edits",
                "review_confidence": "high",
                "entities_final": [
                    {"text": "Ivete Sangalo", "label": "Person", "start": 0, "end": 13},
                    {"text": "Salvador", "label": "Location", "start": 17, "end": 25},
                ],
                "justification": "literal training extraction",
            },
            source_row,
            annotation_mode="train_annotation",
        )
        self.assertEqual(len(validated["entities_final"]), 2)

    def test_validate_adjudication_relocates_unique_occurrence_when_offsets_are_wrong(self):
        source_row = {
            "text": "Tem um carro na tr Miguel pinto em frente ao número 394",
            "review_seed_entities": [],
        }
        validated = validate_adjudication(
            {
                "decision": "accept",
                "review_confidence": "high",
                "entities_final": [
                    {"text": "tr Miguel pinto", "label": "Location", "start": 30, "end": 45},
                ],
                "justification": "unique literal occurrence",
            },
            source_row,
            annotation_mode="train_annotation",
        )
        self.assertEqual(validated["entities_final"][0]["start"], 16)
        self.assertEqual(validated["entities_final"][0]["end"], 31)

    def test_validate_adjudication_rejects_ambiguous_relocation(self):
        source_row = {
            "text": "QUITUNGO perto do posto e depois QUITUNGO novamente",
            "review_seed_entities": [],
        }
        with self.assertRaises(ValueError):
            validate_adjudication(
                {
                    "decision": "accept",
                    "review_confidence": "high",
                    "entities_final": [
                        {"text": "QUITUNGO", "label": "Location", "start": 10, "end": 18},
                    ],
                    "justification": "ambiguous repeated literal occurrence",
                },
                source_row,
                annotation_mode="train_annotation",
            )

    def test_validate_adjudication_rejects_single_token_location_without_agreement_support(self):
        source_row = {
            "text": "rua alagoas proximo deposito de gas",
            "review_seed_entities": [
                {"text": "alagoas", "label": "Location", "seed_origin": "gliner2_location_metadata_match"}
            ],
        }
        with self.assertRaises(ValueError):
            validate_adjudication(
                {
                    "decision": "accept_with_edits",
                    "review_confidence": "high",
                    "entities_final": [{"text": "alagoas", "label": "Location"}],
                    "justification": "single weak location",
                },
                source_row,
            )

    def test_validate_adjudication_rejects_empty_accept_with_edits(self):
        source_row = {
            "text": "CENTRO SÃO JOÃO MERITI",
            "review_seed_entities": [
                {"text": "são joão meriti", "label": "Location", "seed_origin": "baseline_high_score"}
            ],
        }
        with self.assertRaises(ValueError):
            validate_adjudication(
                {
                    "decision": "accept_with_edits",
                    "review_confidence": "high",
                    "entities_final": [],
                    "justification": "removed everything",
                },
                source_row,
            )

    def test_validate_adjudication_allows_multi_location_accept_with_edits(self):
        source_row = {
            "text": "São crias da Coreia. Rua Caruaru escadao.",
            "review_seed_entities": [
                {"text": "Coreia", "label": "Location", "start": 13, "end": 19, "seed_origin": "gliner2_location_metadata_match"},
                {"text": "Caruaru", "label": "Location", "start": 25, "end": 32, "seed_origin": "gliner2_location_metadata_match"},
            ],
        }
        validated = validate_adjudication(
            {
                "decision": "accept_with_edits",
                "review_confidence": "high",
                "entities_final": [
                    {"text": "Coreia", "label": "Location", "start": 13, "end": 19},
                    {"text": "Caruaru", "label": "Location", "start": 25, "end": 32},
                ],
                "justification": "two specific locations",
            },
            source_row,
        )
        self.assertEqual(len(validated["entities_final"]), 2)

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
                    "review_seed_entities": [{"text": "Ivete Sangalo", "label": "Person", "start": 0, "end": 14}],
                },
                model="gpt-4o-mini",
                temperature=0.0,
                annotation_mode="literal_review",
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
