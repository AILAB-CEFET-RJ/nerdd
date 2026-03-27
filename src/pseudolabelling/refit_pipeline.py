import json
import logging
import random
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from base_model_training.io_utils import load_jsonl, save_jsonl
from base_model_training.paths import resolve_path
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
        source = "spans"
        entities = record.get("spans")
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


def prepare_training_records(rows, allowed_labels, text_keys=DEFAULT_TEXT_KEYS, source_name="unknown"):
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
        enriched["_refit_input_meta"] = {
            "text_source": text_source,
            "entity_source": entity_source,
            "training_source": source_name,
        }
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


def build_refit_splits(
    supervised_rows,
    pseudolabel_rows,
    *,
    include_supervised_train,
    include_pseudolabel_train,
    val_ratio,
    seed,
    deduplicate_by_text,
):
    if include_supervised_train and supervised_rows:
        train_supervised_rows, val_rows = split_train_val(supervised_rows, val_ratio=val_ratio, seed=seed)
        train_pseudolabel_rows = pseudolabel_rows if include_pseudolabel_train else []
        train_rows, merge_counts = merge_training_sources(
            train_supervised_rows,
            train_pseudolabel_rows,
            deduplicate_by_text=deduplicate_by_text,
        )
        val_prepare_counts = Counter({"input_records": len(val_rows), "kept_records": len(val_rows)})
        return train_rows, val_rows, merge_counts, val_prepare_counts

    if include_pseudolabel_train:
        train_rows, val_rows = split_train_val(pseudolabel_rows, val_ratio=val_ratio, seed=seed)
        merge_counts = Counter({"kept_pseudolabel": len(train_rows) + len(val_rows), "merged_total": len(train_rows) + len(val_rows)})
        val_prepare_counts = Counter({"input_records": len(val_rows), "kept_records": len(val_rows)})
        return train_rows, val_rows, merge_counts, val_prepare_counts

    raise RuntimeError("No enabled training source is available to build train/validation splits.")


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


def resolve_pseudolabel_input(script_dir, config):
    explicit_path = getattr(config, "pseudolabel_path", "")
    if explicit_path:
        resolved = resolve_path(script_dir, explicit_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Pseudolabel dataset not found: {resolved}")
        return resolved

    input_candidate = resolve_path(script_dir, config.input_path)
    resolved = _pick_input_jsonl(input_candidate)
    if not resolved.exists():
        raise FileNotFoundError(f"Input JSONL not found: {resolved}")
    return resolved


def _load_json_or_jsonl(path):
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return load_jsonl(str(source))

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError(f"Unsupported JSON content in {source}")


def _normalize_text_key(value):
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip().casefold()


def merge_training_sources(supervised_rows, pseudolabel_rows, *, deduplicate_by_text):
    merged = []
    counters = Counter()
    seen_texts = set()

    def _append_rows(rows, source_name):
        for row in rows:
            dedup_key = _normalize_text_key(row.get("text", "")) if deduplicate_by_text else ""
            if deduplicate_by_text and dedup_key:
                if dedup_key in seen_texts:
                    counters[f"dropped_duplicate_{source_name}"] += 1
                    continue
                seen_texts.add(dedup_key)
            merged.append(row)
            counters[f"kept_{source_name}"] += 1

    _append_rows(supervised_rows, "supervised")
    _append_rows(pseudolabel_rows, "pseudolabel")
    counters["merged_total"] = len(merged)
    return merged, counters


def _resolve_refit_sources(config):
    mode = getattr(config, "refit_mode", "supervised_plus_pseudolabels")
    if mode == "supervised_only":
        return True, False
    if mode == "supervised_plus_pseudolabels":
        return True, True
    if mode == "pseudolabel_only":
        return False, True
    raise ValueError(f"Unsupported refit_mode: {mode}")


def sample_pseudolabel_rows(rows, *, sample_ratio, max_records, seed):
    sampled = list(rows)
    counters = Counter(
        {
            "input_records": len(rows),
            "sample_ratio": float(sample_ratio),
            "max_records": int(max_records),
        }
    )

    if sampled and sample_ratio < 1.0:
        keep_count = max(1, int(round(len(sampled) * sample_ratio)))
        rng = random.Random(seed)
        sampled = rng.sample(sampled, keep_count)
        counters["dropped_by_ratio"] = len(rows) - len(sampled)

    if sampled and max_records > 0 and len(sampled) > max_records:
        rng = random.Random(seed + 1)
        sampled = rng.sample(sampled, max_records)
        counters["dropped_by_cap"] = counters["input_records"] - counters["dropped_by_ratio"] - len(sampled)

    counters["kept_records"] = len(sampled)
    return sampled, counters


def run_refit(config, script_path):
    started_at = datetime.now(timezone.utc).isoformat()
    timer = perf_counter()
    set_seed(config.seed)

    script_dir = Path(script_path).resolve().parent
    output_model_dir = resolve_path(script_dir, config.output_model_dir)
    stats_json = resolve_path(script_dir, config.stats_json)
    train_manifest_jsonl = resolve_path(script_dir, config.train_manifest_jsonl)
    val_manifest_jsonl = resolve_path(script_dir, config.val_manifest_jsonl)

    input_jsonl = resolve_pseudolabel_input(script_dir, config)

    supervised_json = None
    if config.supervised_train_path:
        supervised_json = resolve_path(script_dir, config.supervised_train_path)
        if not supervised_json.exists():
            raise FileNotFoundError(f"Supervised training dataset not found: {supervised_json}")

    val_jsonl = None
    if config.val_jsonl:
        val_jsonl = resolve_path(script_dir, config.val_jsonl)
        if not val_jsonl.exists():
            raise FileNotFoundError(f"Validation JSONL not found: {val_jsonl}")

    output_model_dir.mkdir(parents=True, exist_ok=True)
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    train_manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)
    val_manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)

    raw_pseudolabel_rows = load_jsonl(str(input_jsonl))
    prepared_pseudolabel_rows, pseudolabel_prepare_counts = prepare_training_records(
        raw_pseudolabel_rows,
        allowed_labels=set(config.allowed_labels),
        source_name="pseudolabel",
    )
    pseudolabel_sample_counts = Counter(
        {
            "input_records": len(prepared_pseudolabel_rows),
            "kept_records": len(prepared_pseudolabel_rows),
            "sample_ratio": float(config.pseudolabel_sample_ratio),
            "max_records": int(config.max_pseudolabel_records),
        }
    )
    if prepared_pseudolabel_rows and (
        config.pseudolabel_sample_ratio < 1.0 or config.max_pseudolabel_records > 0
    ):
        prepared_pseudolabel_rows, pseudolabel_sample_counts = sample_pseudolabel_rows(
            prepared_pseudolabel_rows,
            sample_ratio=config.pseudolabel_sample_ratio,
            max_records=config.max_pseudolabel_records,
            seed=config.seed,
        )

    include_supervised_train, include_pseudolabel_train = _resolve_refit_sources(config)

    prepared_supervised_rows = []
    supervised_prepare_counts = Counter()
    if include_supervised_train:
        if supervised_json is None:
            raise RuntimeError(
                "Refit mode requires supervised data, but no supervised_train_path was provided."
            )
        raw_supervised_rows = _load_json_or_jsonl(supervised_json)
        prepared_supervised_rows, supervised_prepare_counts = prepare_training_records(
            raw_supervised_rows,
            allowed_labels=set(config.allowed_labels),
            source_name="supervised",
        )
        if not prepared_supervised_rows:
            raise RuntimeError("Supervised training dataset provided, but no valid records were found.")

    if not include_pseudolabel_train:
        prepared_pseudolabel_rows = []
        pseudolabel_prepare_counts = Counter({"skipped_by_refit_mode": 1, "input_records": len(raw_pseudolabel_rows)})

    if val_jsonl:
        raw_val_rows = _load_json_or_jsonl(val_jsonl)
        val_rows, val_prepare_counts = prepare_training_records(
            raw_val_rows,
            allowed_labels=set(config.allowed_labels),
            source_name="validation",
        )
        if not val_rows:
            raise RuntimeError("Validation file provided, but no valid validation records were found.")
        train_rows, merge_counts = merge_training_sources(
            prepared_supervised_rows,
            prepared_pseudolabel_rows,
            deduplicate_by_text=config.deduplicate_by_text,
        )
    else:
        train_rows, val_rows, merge_counts, val_prepare_counts = build_refit_splits(
            prepared_supervised_rows,
            prepared_pseudolabel_rows,
            include_supervised_train=include_supervised_train,
            include_pseudolabel_train=include_pseudolabel_train,
            val_ratio=config.val_ratio,
            seed=config.seed,
            deduplicate_by_text=config.deduplicate_by_text,
        )

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
        max_length=int(config.max_length),
        overlap=int(config.overlap),
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
            "pseudolabel_path": getattr(config, "pseudolabel_path", ""),
            "supervised_train_path": config.supervised_train_path,
            "refit_mode": config.refit_mode,
            "input_jsonl_resolved": str(input_jsonl.resolve()),
            "supervised_train_resolved": str(supervised_json.resolve()) if supervised_json else None,
            "output_model_dir": config.output_model_dir,
            "base_model": base_model,
            "epochs": config.epochs,
            "patience": config.patience,
            "batch_size": config.batch_size,
            "max_length": config.max_length,
            "overlap": config.overlap,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "val_jsonl": config.val_jsonl,
            "val_ratio": config.val_ratio,
            "seed": config.seed,
            "allowed_labels": config.allowed_labels,
            "num_workers": config.num_workers,
            "include_supervised_train": config.include_supervised_train,
            "deduplicate_by_text": config.deduplicate_by_text,
            "pseudolabel_sample_ratio": config.pseudolabel_sample_ratio,
            "max_pseudolabel_records": config.max_pseudolabel_records,
            "effective_include_supervised_train": include_supervised_train,
            "effective_include_pseudolabel_train": include_pseudolabel_train,
        },
        "data_summary": {
            "prepare_counts_pseudolabel_source": dict(pseudolabel_prepare_counts),
            "sample_counts_pseudolabel_source": dict(pseudolabel_sample_counts),
            "prepare_counts_supervised_source": dict(supervised_prepare_counts),
            "prepare_counts_val_source": dict(val_prepare_counts),
            "merge_counts": dict(merge_counts),
            "train_size": len(train_rows),
            "val_size": len(val_rows),
            "train_source_breakdown": dict(
                Counter(row.get("_refit_input_meta", {}).get("training_source", "unknown") for row in train_rows)
            ),
            "val_source_breakdown": dict(
                Counter(row.get("_refit_input_meta", {}).get("training_source", "unknown") for row in val_rows)
            ),
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
