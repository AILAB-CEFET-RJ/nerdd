import json
import logging
import random
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from gliner_train.io_utils import load_jsonl, save_jsonl
from gliner_train.paths import resolve_path
from pseudolabelling.refit_backend import load_gliner, treinar_gliner

LOGGER = logging.getLogger(__name__)
DEFAULT_TEXT_KEYS = ("text", "relato", "texto", "descricao", "description")


def set_seed(seed):
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_text(record, text_keys=DEFAULT_TEXT_KEYS):
    for key in text_keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), key
    return "", ""


def normalize_entities(record, allowed_labels):
    source = "entities"
    entities = record.get("entities")
    if not isinstance(entities, list):
        source = "ner"
        entities = record.get("ner")
    if not isinstance(entities, list):
        return [], source, Counter({"missing_entity_list": 1})

    counters = Counter()
    normalized = []
    for entity in entities:
        start = _safe_int(entity.get("start"))
        end = _safe_int(entity.get("end"))
        label = entity.get("label")
        if not isinstance(label, str) or not label.strip():
            counters["invalid_label"] += 1
            continue
        label = label.strip()
        if allowed_labels and label not in allowed_labels:
            counters["filtered_by_label"] += 1
            continue
        if start is None or end is None or end <= start:
            counters["invalid_span"] += 1
            continue
        normalized.append({"start": start, "end": end, "label": label})

    if not normalized:
        counters["no_valid_entities"] += 1
    return normalized, source, counters


def prepare_training_records(rows, allowed_labels, text_keys=DEFAULT_TEXT_KEYS):
    prepared = []
    counters = Counter()
    for row in rows:
        text_value, text_source = extract_text(row, text_keys=text_keys)
        if not text_value:
            counters["missing_text"] += 1
            continue
        entities, entity_source, entity_counters = normalize_entities(row, allowed_labels=allowed_labels)
        counters.update(entity_counters)
        if not entities:
            counters["dropped_no_entities"] += 1
            continue
        enriched = deepcopy(row)
        enriched["text"] = text_value
        enriched["entities"] = entities
        enriched["_refit_input_meta"] = {"text_source": text_source, "entity_source": entity_source}
        prepared.append(enriched)
        counters["kept_records"] += 1
    counters["input_records"] = len(rows)
    return prepared, counters


def split_train_val(records, val_ratio, seed):
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if not shuffled:
        return [], []
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    val_records = shuffled[:n_val]
    train_records = shuffled[n_val:]
    if not train_records:
        train_records, val_records = shuffled, []
    return train_records, val_records


def _format_duration(seconds):
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def _pick_input_jsonl(input_path):
    if input_path.is_file():
        return input_path
    candidates = ["kept.jsonl", "mantidos.jsonl", "preds002_iso_mantidos.jsonl"]
    for name in candidates:
        candidate = input_path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No kept file found inside {input_path}. Tried: {', '.join(candidates)}"
    )


def run_refit(config, script_path):
    started_at = datetime.now(timezone.utc).isoformat()
    timer = perf_counter()
    set_seed(config.seed)

    script_dir = Path(script_path).resolve().parent
    input_candidate = resolve_path(script_dir, config.input_path)
    output_model_dir = resolve_path(script_dir, config.output_model_dir)
    stats_json = resolve_path(script_dir, config.stats_json)
    train_manifest_jsonl = resolve_path(script_dir, config.train_manifest_jsonl)
    val_manifest_jsonl = resolve_path(script_dir, config.val_manifest_jsonl)

    input_jsonl = _pick_input_jsonl(input_candidate)
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    val_jsonl = None
    if config.val_jsonl:
        val_jsonl = resolve_path(script_dir, config.val_jsonl)
        if not val_jsonl.exists():
            raise FileNotFoundError(f"Validation JSONL not found: {val_jsonl}")

    output_model_dir.mkdir(parents=True, exist_ok=True)
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    train_manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)
    val_manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)

    raw_rows = load_jsonl(str(input_jsonl))
    prepared_rows, prepare_counts = prepare_training_records(
        raw_rows,
        allowed_labels=set(config.allowed_labels),
    )
    if not prepared_rows:
        raise RuntimeError("No valid training records after preprocessing.")

    if val_jsonl:
        raw_val_rows = load_jsonl(str(val_jsonl))
        val_rows, val_prepare_counts = prepare_training_records(
            raw_val_rows,
            allowed_labels=set(config.allowed_labels),
        )
        if not val_rows:
            raise RuntimeError("Validation file provided, but no valid validation records were found.")
        train_rows = prepared_rows
    else:
        train_rows, val_rows = split_train_val(prepared_rows, val_ratio=config.val_ratio, seed=config.seed)
        val_prepare_counts = Counter({"input_records": len(val_rows), "kept_records": len(val_rows)})

    if not train_rows:
        raise RuntimeError("No training records available for refit.")

    save_jsonl(str(train_manifest_jsonl), train_rows)
    save_jsonl(str(val_manifest_jsonl), val_rows)

    base_model = config.base_model
    if not base_model:
        for candidate in ("best_overall_gliner_model", "best_overall_gliner"):
            project_candidate = script_dir.parent / candidate
            if project_candidate.exists():
                base_model = str(project_candidate)
                break
        if not base_model:
            base_model = "best_overall_gliner_model"

    LOGGER.info("Loading base model for refit: %s", base_model)
    model, _tokenizer, _device = load_gliner(base_model)

    LOGGER.info(
        "Starting refit | train=%s val=%s epochs=%s patience=%s batch_size=%s lr=%s wd=%s",
        len(train_rows),
        len(val_rows),
        config.epochs,
        config.patience,
        config.batch_size,
        config.lr,
        config.weight_decay,
    )
    treinar_gliner(
        model=model,
        train_recs=train_rows,
        val_recs=val_rows,
        tokenizer=None,
        out_dir=str(output_model_dir),
        num_epochs=int(config.epochs),
        paciencia=int(config.patience),
        batch_size=int(config.batch_size),
        lr=float(config.lr),
        wd=float(config.weight_decay),
        dl_num_workers=int(config.num_workers),
    )

    finished_at = datetime.now(timezone.utc).isoformat()
    runtime_seconds = perf_counter() - timer
    stats_payload = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_seconds": runtime_seconds,
        "runtime_hms": _format_duration(runtime_seconds),
        "config": {
            "input_path": config.input_path,
            "input_jsonl_resolved": str(input_jsonl.resolve()),
            "output_model_dir": config.output_model_dir,
            "base_model": base_model,
            "epochs": config.epochs,
            "patience": config.patience,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "val_jsonl": config.val_jsonl,
            "val_ratio": config.val_ratio,
            "seed": config.seed,
            "allowed_labels": config.allowed_labels,
            "num_workers": config.num_workers,
        },
        "data_summary": {
            "prepare_counts_train_source": dict(prepare_counts),
            "prepare_counts_val_source": dict(val_prepare_counts),
            "train_size": len(train_rows),
            "val_size": len(val_rows),
        },
        "artifacts": {
            "model_dir": str(output_model_dir.resolve()),
            "train_manifest_jsonl": str(train_manifest_jsonl.resolve()),
            "val_manifest_jsonl": str(val_manifest_jsonl.resolve()),
        },
    }
    with open(stats_json, "w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, indent=2, ensure_ascii=False)

    LOGGER.info("Refit completed. Model dir: %s", output_model_dir)
    LOGGER.info("Refit stats saved to: %s", stats_json)
