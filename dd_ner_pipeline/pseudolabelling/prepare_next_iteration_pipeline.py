import glob
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from gliner_train.io_utils import save_jsonl
from gliner_train.paths import resolve_path

LOGGER = logging.getLogger(__name__)


def _iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSONL at line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_no} is not a JSON object")
            yield row


def _load_records(path, allow_json=False):
    if not allow_json:
        return list(_iter_jsonl(path))

    text = Path(path).read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return list(_iter_jsonl(path))

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        out = []
        for idx, item in enumerate(parsed, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"{path}: JSON list item {idx} is not an object")
            out.append(item)
        return out
    raise ValueError(f"{path}: unsupported JSON content (expected object or list)")


def _coerce_value(value, policy, fill_missing_with, field, path):
    if value is None:
        return fill_missing_with
    if isinstance(value, str):
        return value
    if policy == "stringify":
        return str(value)
    if policy == "empty":
        return fill_missing_with
    raise ValueError(f"{path}: field '{field}' has non-string value and coerce policy is 'error'")


def _project_record(record, keep_fields, *, coerce_non_string, fill_missing_with, source_path):
    projected = {}
    for field in keep_fields:
        value = record.get(field, fill_missing_with)
        projected[field] = _coerce_value(
            value,
            policy=coerce_non_string,
            fill_missing_with=fill_missing_with,
            field=field,
            path=source_path,
        )
    return projected


def _is_non_empty(value):
    return isinstance(value, str) and bool(value.strip())


def process_records(records, *, keep_fields, required_fields, fill_missing_with, coerce_non_string, drop_empty_relato, deduplicate_by, source_path):
    output = []
    counters = Counter()
    seen = set()

    for record in records:
        counters["input_rows"] += 1
        projected = _project_record(
            record,
            keep_fields,
            coerce_non_string=coerce_non_string,
            fill_missing_with=fill_missing_with,
            source_path=source_path,
        )

        missing_required = [field for field in required_fields if not _is_non_empty(projected.get(field, ""))]
        if missing_required:
            counters["dropped_missing_required"] += 1
            continue

        if drop_empty_relato and not _is_non_empty(projected.get("relato", "")):
            counters["dropped_empty_relato"] += 1
            continue

        if deduplicate_by:
            dedup_key = tuple(projected.get(field, "") for field in deduplicate_by)
            if dedup_key in seen:
                counters["dropped_duplicate"] += 1
                continue
            seen.add(dedup_key)

        counters["output_rows"] += 1
        output.append(projected)

    return output, counters


def _resolve_io_pairs(config, script_dir):
    if config["input_jsonl"]:
        input_path = resolve_path(script_dir, config["input_jsonl"])
        output_path = resolve_path(script_dir, config["output_jsonl"])
        return [(input_path, output_path)]

    pattern = str(resolve_path(script_dir, config["input_glob"]))
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched input glob: {pattern}")
    out_dir = resolve_path(script_dir, config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for match in matches:
        src = Path(match)
        dst = out_dir / f"{src.stem}{config['output_suffix']}.jsonl"
        pairs.append((src, dst))
    return pairs


def _format_duration(seconds):
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def run_prepare_next_iteration(config, script_path):
    started_at = datetime.now(timezone.utc).isoformat()
    timer = perf_counter()
    script_dir = Path(script_path).resolve().parent

    io_pairs = _resolve_io_pairs(config, script_dir)
    stats_json = resolve_path(script_dir, config["stats_json"])
    stats_json.parent.mkdir(parents=True, exist_ok=True)

    per_file = []
    total = Counter()

    for input_path, output_path in io_pairs:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        records = _load_records(str(input_path), allow_json=config["allow_json"])
        processed, counters = process_records(
            records,
            keep_fields=config["keep_fields"],
            required_fields=config["required_fields"],
            fill_missing_with=config["fill_missing_with"],
            coerce_non_string=config["coerce_non_string"],
            drop_empty_relato=config["drop_empty_relato"],
            deduplicate_by=config["deduplicate_by"],
            source_path=str(input_path),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_jsonl(str(output_path), processed)
        total.update(counters)
        per_file.append(
            {
                "input": str(input_path.resolve()),
                "output": str(output_path.resolve()),
                "counts": dict(counters),
            }
        )
        LOGGER.info(
            "Prepared next-iteration file: %s -> %s | input=%s output=%s",
            input_path.name,
            output_path.name,
            counters["input_rows"],
            counters["output_rows"],
        )

    runtime_seconds = perf_counter() - timer
    finished_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_seconds": runtime_seconds,
        "runtime_hms": _format_duration(runtime_seconds),
        "config": config,
        "summary": dict(total),
        "files": per_file,
    }
    with open(stats_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Saved prepare-next-iteration stats to: %s", stats_json)
