#!/usr/bin/env python3
"""Run LLM-assisted adjudication over structured NER candidate review inputs.

This utility consumes the JSONL produced by `build_llm_adjudication_input.py`,
calls the OpenAI Responses API with Structured Outputs, and writes a per-record
JSONL containing the model's adjudication decision plus the original source row.

It is intentionally narrow:
- input rows already contain baseline/GLiNER2 suggestions and matching metadata
- the LLM must only decide among `accept`, `accept_with_edits`, or `reject`
- the LLM must return only literal spans present in the input text
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_model_training.paths import resolve_repo_artifact_path
from tools.inspect_dense_tips import read_json_or_jsonl, write_jsonl

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5"
DEFAULT_API_BASE = "https://api.openai.com/v1/responses"
DEFAULT_ALLOWED_LABELS = ["Person", "Location", "Organization"]
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_DOTENV_PATH = ".env"
DEFAULT_TEMPERATURE = 0.0
ALLOWED_DECISIONS = {"accept", "accept_with_edits", "reject"}
ALLOWED_LABELS = set(DEFAULT_ALLOWED_LABELS)

ADJUDICATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["accept", "accept_with_edits", "reject"],
        },
        "review_confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "entities_final": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "label": {
                        "type": "string",
                        "enum": list(DEFAULT_ALLOWED_LABELS),
                    },
                },
                "required": ["text", "label"],
            },
        },
        "justification": {"type": "string"},
    },
    "required": ["decision", "review_confidence", "entities_final", "justification"],
}


class AdjudicationValidationError(ValueError):
    """Raised when model output violates semantic adjudication constraints."""


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collapse_spaces(text: str) -> str:
    return " ".join(str(text).split())


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_entity_text(text: str) -> str:
    text = _collapse_spaces(str(text).strip().lower())
    text = _strip_accents(text)
    return text.strip(" \t\r\n.,;:!?()[]{}\"'")


def _normalized_tokens(text: str) -> list[str]:
    normalized = normalize_entity_text(text)
    if not normalized:
        return []
    return [token for token in normalized.split() if token]


def _review_seed_index(source_row: dict) -> dict[tuple[str, str], dict]:
    return {
        (str(item.get("text", "")), str(item.get("label", ""))): item
        for item in (source_row.get("review_seed_entities") or [])
        if isinstance(item, dict)
    }


def _is_weak_single_location_accept(entities: list[dict], source_row: dict) -> bool:
    if len(entities) != 1:
        return False
    entity = entities[0]
    if entity.get("label") != "Location":
        return False
    if len(_normalized_tokens(entity.get("text", ""))) != 1:
        return False
    seed = _review_seed_index(source_row).get((entity["text"], entity["label"]), {})
    seed_origin = str(seed.get("seed_origin", ""))
    return not seed_origin.startswith("agreed_")


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def resolve_api_key(*, env_name: str, dotenv_values: dict[str, str]) -> str:
    return os.environ.get(env_name, "").strip() or dotenv_values.get(env_name, "").strip()


def resolve_model_name(cli_value: str, dotenv_values: dict[str, str]) -> str:
    cli_value = str(cli_value or "").strip()
    if cli_value:
        return cli_value
    return dotenv_values.get("OPENAI_DEFAULT_MODEL", "").strip() or DEFAULT_MODEL


def resolve_temperature(cli_value, dotenv_values: dict[str, str]) -> float:
    if cli_value is not None:
        parsed = _safe_float(cli_value)
        if parsed is None:
            raise ValueError(f"Invalid CLI temperature: {cli_value}")
        return parsed
    env_value = dotenv_values.get("OPENAI_DEFAULT_TEMPERATURE", "").strip()
    if env_value:
        parsed = _safe_float(env_value)
        if parsed is None:
            raise ValueError(f"Invalid OPENAI_DEFAULT_TEMPERATURE in .env: {env_value}")
        return parsed
    return DEFAULT_TEMPERATURE


def model_supports_temperature(model: str) -> bool:
    model_name = str(model or "").strip().lower()
    if model_name.startswith("gpt-5") and not model_name.startswith("gpt-5.1"):
        return False
    return True


def _compact_entity(entity: dict) -> dict:
    compact = {
        "text": entity.get("text", ""),
        "label": entity.get("label", ""),
    }
    ner_score = _safe_float(entity.get("ner_score"))
    if ner_score is not None:
        compact["ner_score"] = round(ner_score, 6)
    seed_origin = entity.get("seed_origin")
    if seed_origin:
        compact["seed_origin"] = seed_origin
    match_type = entity.get("match_type")
    if match_type:
        compact["match_type"] = match_type
    return compact


def _serialize_entities(entities) -> list[dict]:
    rows = []
    for entity in entities or []:
        if isinstance(entity, dict):
            rows.append(_compact_entity(entity))
    return rows


def build_messages(row: dict) -> list[dict]:
    text = str(row.get("text", "")).strip()
    payload = {
        "source_id": row.get("source_id"),
        "record_score": row.get("record_score"),
        "candidate_quality_score": row.get("candidate_quality_score"),
        "metadata": row.get("metadata", {}),
        "baseline_entities": _serialize_entities(row.get("baseline_entities")),
        "gliner2_entities": _serialize_entities(row.get("gliner2_entities")),
        "agreed_entities": [
            {
                "text": item.get("text", ""),
                "label": item.get("label", ""),
                "match_type": item.get("match_type", ""),
                "consensus_score": item.get("consensus_score"),
            }
            for item in (row.get("agreed_entities") or [])
            if isinstance(item, dict)
        ],
        "baseline_only_entities": _serialize_entities(row.get("baseline_only_entities")),
        "gliner2_only_entities": _serialize_entities(row.get("gliner2_only_entities")),
        "conflicts": [
            {
                "text": item.get("text", ""),
                "baseline_label": item.get("baseline_label", ""),
                "gliner2_label": item.get("gliner2_label", ""),
                "conflict_type": item.get("conflict_type", ""),
            }
            for item in (row.get("conflicts") or [])
            if isinstance(item, dict)
        ],
        "review_seed_entities": _serialize_entities(row.get("review_seed_entities")),
    }

    system_message = (
        "You are a reviewer of Portuguese NER annotations for Brazilian crime-tip records.\n"
        "Return JSON only.\n"
        "Domain context:\n"
        "- Each record is an anonymous Brazilian crime tip.\n"
        "- Text is often noisy, non-standard, misspelled, truncated, and may contain OCR/encoding corruption.\n"
        "- Conservatism is required.\n"
        "Hard rules:\n"
        "1. Every entity text must be an exact literal substring of the input text.\n"
        "2. Do not normalize, simplify, shorten, translate, or correct spelling in entity text.\n"
        "3. Allowed labels: Person, Location, Organization.\n"
        "4. Prefer reject over speculative completion.\n"
        "5. decision='accept' is only allowed when every final entity comes directly from review_seed_entities.\n"
        "6. decision='accept_with_edits' is only allowed when every final entity comes directly from review_seed_entities.\n"
        "7. For accept_with_edits, you may remove entities from the seed set, but do not add new entities outside review_seed_entities.\n"
        "8. Do not promote gliner2-only entities unless they are already present in review_seed_entities.\n"
        "9. If the case is noisy, incomplete, ambiguous, weakly supported, or semantically poor, return decision='reject'.\n"
        "10. Be especially skeptical of single-token Location entities, partial place names, generic road/area words, and corrupted fragments.\n"
        "11. If in doubt, reject."
    )
    user_message = (
        "Review this single Brazilian crime-tip record and adjudicate the proposed NER annotation.\n\n"
        "Task:\n"
        "- Decide whether the review_seed_entities for this record should be accepted as-is, accepted with removals, or rejected.\n"
        "- Do not perform open-ended entity extraction.\n"
        "- Only keep entities that are exact literal substrings of the TEXT and already present in review_seed_entities.\n"
        "- If the seed set is noisy, weak, generic, partial, corrupted, or ambiguous, reject.\n\n"
        f"TEXT:\n{text}\n\n"
        "CANDIDATE DATA:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def build_request_body(row: dict, *, model: str, temperature: float | None = None) -> dict:
    body = {
        "model": model,
        "input": build_messages(row),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ner_adjudication",
                "strict": True,
                "schema": ADJUDICATION_SCHEMA,
            }
        },
    }
    if temperature is not None and model_supports_temperature(model):
        body["temperature"] = temperature
    return body


def _extract_output_text(response_payload: dict) -> str:
    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = response_payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()

    raise AdjudicationValidationError("Could not extract model output text from Responses API payload.")


def parse_adjudication_response(response_payload: dict) -> dict:
    output_text = _extract_output_text(response_payload)
    parsed = json.loads(output_text)
    if not isinstance(parsed, dict):
        raise AdjudicationValidationError("Structured output is not a JSON object.")
    return parsed


def validate_adjudication(adjudication: dict, source_row: dict) -> dict:
    if not isinstance(adjudication, dict):
        raise AdjudicationValidationError("Adjudication payload must be a JSON object.")

    decision = adjudication.get("decision")
    if decision not in ALLOWED_DECISIONS:
        raise AdjudicationValidationError(f"Invalid adjudication decision: {decision}")

    entities = adjudication.get("entities_final")
    if not isinstance(entities, list):
        raise AdjudicationValidationError("entities_final must be a list.")

    text = str(source_row.get("text", ""))
    review_seed_entities = source_row.get("review_seed_entities") or []
    review_seed_pairs = {
        (str(item.get("text", "")), str(item.get("label", "")))
        for item in review_seed_entities
        if isinstance(item, dict)
    }

    cleaned_entities = []
    seen = set()
    for entity in entities:
        if not isinstance(entity, dict):
            raise AdjudicationValidationError("Each entity in entities_final must be an object.")
        entity_text = str(entity.get("text", ""))
        entity_label = str(entity.get("label", ""))
        if not entity_text or entity_label not in ALLOWED_LABELS:
            raise AdjudicationValidationError(f"Invalid entity returned by adjudicator: {entity}")
        if entity_text not in text:
            raise AdjudicationValidationError(f"Entity text is not an exact substring of the source text: {entity_text!r}")
        key = (entity_text, entity_label)
        if key in seen:
            continue
        seen.add(key)
        cleaned_entities.append({"text": entity_text, "label": entity_label})

    if decision in {"accept", "accept_with_edits"}:
        if not cleaned_entities:
            raise AdjudicationValidationError(
                f"decision={decision!r} must contain at least one final entity"
            )
        invalid_seed_entities = [entity for entity in cleaned_entities if (entity["text"], entity["label"]) not in review_seed_pairs]
        if invalid_seed_entities:
            raise AdjudicationValidationError(
                f"decision={decision!r} may only contain entities from review_seed_entities; "
                f"got unsupported entities: {invalid_seed_entities}"
            )
        if _is_weak_single_location_accept(cleaned_entities, source_row):
            raise AdjudicationValidationError(
                f"decision={decision!r} is too weak semantically: single-token Location without agreement support"
            )

    validated = dict(adjudication)
    validated["entities_final"] = cleaned_entities
    return validated


def call_responses_api(
    row: dict,
    *,
    model: str,
    temperature: float | None,
    api_key: str,
    api_base: str,
    timeout_seconds: int,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = build_request_body(row, model=model, temperature=temperature)
    response = requests.post(api_base, headers=headers, json=body, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def adjudicate_row(
    row: dict,
    *,
    model: str,
    temperature: float | None,
    api_key: str,
    api_base: str,
    timeout_seconds: int,
    max_retries: int,
    retry_sleep_seconds: float,
) -> dict:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response_payload = call_responses_api(
                row,
                model=model,
                temperature=temperature,
                api_key=api_key,
                api_base=api_base,
                timeout_seconds=timeout_seconds,
            )
            adjudication = validate_adjudication(parse_adjudication_response(response_payload), row)
            return {
                "source_id": row.get("source_id"),
                "model": model,
                "temperature": temperature,
                "adjudication": adjudication,
                "_source": row,
            }
        except AdjudicationValidationError as exc:
            LOGGER.warning(
                "Validation downgrade to reject for source_id=%s on attempt %s/%s: %s",
                row.get("source_id"),
                attempt,
                max_retries,
                exc,
            )
            return {
                "source_id": row.get("source_id"),
                "model": model,
                "temperature": temperature,
                "adjudication": {
                    "decision": "reject",
                    "review_confidence": "low",
                    "entities_final": [],
                    "justification": f"Validation downgrade: {exc}",
                },
                "_source": row,
            }
        except Exception as exc:  # pragma: no cover - exercised in integration
            last_error = exc
            LOGGER.warning(
                "Adjudication failed for source_id=%s on attempt %s/%s: %s",
                row.get("source_id"),
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds)
    raise RuntimeError(f"Adjudication failed for source_id={row.get('source_id')}: {last_error}") from last_error


def build_summary(success_rows: list[dict], error_rows: list[dict], *, model: str, input_path: str, output_path: str) -> dict:
    decisions = {}
    for row in success_rows:
        decision = (((row.get("adjudication") or {}).get("decision")) or "")
        decisions[decision] = decisions.get(decision, 0) + 1
    return {
        "input": input_path,
        "output_jsonl": output_path,
        "model": model,
        "records_total": len(success_rows) + len(error_rows),
        "records_succeeded": len(success_rows),
        "records_failed": len(error_rows),
        "decision_counts": decisions,
    }


def _append_jsonl_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def _load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = read_json_or_jsonl(path)
    return [row for row in rows if isinstance(row, dict)]


def _processed_source_ids(success_rows: list[dict], error_rows: list[dict]) -> set[str]:
    processed: set[str] = set()
    for row in success_rows + error_rows:
        source_id = row.get("source_id")
        if source_id is not None:
            processed.add(str(source_id))
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-assisted adjudication over pseudolabel candidate review inputs.")
    parser.add_argument("--input", required=True, help="Input JSON/JSONL from build_llm_adjudication_input.py.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL with adjudication results.")
    parser.add_argument("--errors-jsonl", default="", help="Optional JSONL with per-record errors.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON.")
    parser.add_argument("--model", default="", help="OpenAI model name. Defaults to OPENAI_DEFAULT_MODEL from .env, then gpt-5.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature. Defaults to OPENAI_DEFAULT_TEMPERATURE from .env, then 0.0.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable holding the OpenAI API key.")
    parser.add_argument("--dotenv-path", default=DEFAULT_DOTENV_PATH, help="Path to the .env file used to load the API key.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="Responses API endpoint.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap on input rows.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    dotenv_values = load_dotenv(resolve_repo_artifact_path(__file__, args.dotenv_path))
    api_key = resolve_api_key(env_name=args.api_key_env, dotenv_values=dotenv_values)
    if not api_key:
        raise RuntimeError(
            f"Missing API key '{args.api_key_env}' in environment or .env file: {args.dotenv_path}"
        )
    model_name = resolve_model_name(args.model, dotenv_values)
    temperature = resolve_temperature(args.temperature, dotenv_values)
    LOGGER.info(
        "Resolved adjudication config: model=%s temperature=%s dotenv=%s",
        model_name,
        temperature,
        args.dotenv_path,
    )

    input_path = resolve_repo_artifact_path(__file__, args.input)
    output_path = resolve_repo_artifact_path(__file__, args.output_jsonl)
    errors_path = resolve_repo_artifact_path(__file__, args.errors_jsonl) if args.errors_jsonl else None

    rows = read_json_or_jsonl(input_path)
    if args.max_records > 0:
        rows = rows[: args.max_records]

    success_rows = _load_existing_rows(output_path)
    error_rows = _load_existing_rows(errors_path) if errors_path else []
    processed_source_ids = _processed_source_ids(success_rows, error_rows)
    remaining_rows = [
        row for row in rows if str(row.get("source_id")) not in processed_source_ids
    ]

    if processed_source_ids:
        LOGGER.info(
            "Resume mode: loaded %s completed rows and %s error rows; %s/%s rows remaining",
            len(success_rows),
            len(error_rows),
            len(remaining_rows),
            len(rows),
        )

    for index, row in enumerate(remaining_rows, start=1):
        try:
            result = adjudicate_row(
                row,
                model=model_name,
                temperature=temperature,
                api_key=api_key,
                api_base=args.api_base,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                retry_sleep_seconds=args.retry_sleep_seconds,
            )
            success_rows.append(result)
            _append_jsonl_row(output_path, result)
            LOGGER.info(
                "Adjudicated %s/%s remaining; total completed=%s source_id=%s",
                index,
                len(remaining_rows),
                len(success_rows),
                row.get("source_id"),
            )
        except Exception as exc:  # pragma: no cover - exercised in integration
            error_row = {
                "source_id": row.get("source_id"),
                "error": str(exc),
                "_source": row,
            }
            error_rows.append(error_row)
            if errors_path:
                _append_jsonl_row(errors_path, error_row)
            LOGGER.error("Failed adjudication for source_id=%s: %s", row.get("source_id"), exc)

    LOGGER.info("Saved adjudication JSONL: %s", args.output_jsonl)

    if args.errors_jsonl:
        LOGGER.info("Saved adjudication errors JSONL: %s", args.errors_jsonl)

    if args.summary_json:
        summary_path = resolve_repo_artifact_path(__file__, args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                build_summary(
                    success_rows,
                    error_rows,
                    model=model_name,
                    input_path=str(input_path),
                    output_path=str(output_path),
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        LOGGER.info("Saved adjudication summary JSON: %s", args.summary_json)


if __name__ == "__main__":
    main()
