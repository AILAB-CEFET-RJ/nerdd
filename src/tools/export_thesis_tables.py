#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export consolidated experiment tables for dissertation writing."
    )
    parser.add_argument(
        "--base-root",
        default="./artifacts/base_model_training",
        help="Root directory containing base-model-training artifacts.",
    )
    parser.add_argument(
        "--pseudolabelling-root",
        default="./artifacts/pseudolabelling",
        help="Root directory containing pseudolabelling artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="./artifacts/dissertation_exports",
        help="Directory where CSV/Markdown exports will be written.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_get(mapping, *path, default=None):
    current = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def stringify(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown_table(path: Path, rows, fieldnames, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join("---" for _ in fieldnames) + " |")
    for row in rows:
        lines.append("| " + " | ".join(stringify(row.get(key, "")) for key in fieldnames) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def experiment_id_from_path(path: Path, anchor: str):
    try:
        relative = path.relative_to(anchor)
        return str(relative.parent).replace("/", "__")
    except Exception:
        return path.parent.name


def collect_nested_cv_rows(base_root: Path):
    rows = []
    if not base_root.exists():
        return rows

    for path in sorted(base_root.rglob("nested_cv_results.json")):
        payload = load_json(path)
        config = payload.get("config", {})
        summary = payload.get("summary", {})
        rows.append(
            {
                "experiment_id": experiment_id_from_path(path, base_root),
                "family": "base_nested_cv",
                "artifact_path": str(path),
                "backbone": config.get("model_base", ""),
                "keep_empty_samples": config.get("keep_empty_samples", False),
                "best_overall_test_f1": summary.get("best_overall_test_f1"),
                "mean_test_f1": summary.get("mean_test_f1"),
                "std_test_f1": summary.get("std_test_f1"),
                "mean_seen_entity_test_f1": summary.get("mean_seen_entity_test_f1"),
                "mean_unseen_entity_test_f1": summary.get("mean_unseen_entity_test_f1"),
                "runtime_hms": summary.get("runtime_hms"),
                "runtime_seconds": summary.get("runtime_seconds"),
                "status": "final",
                "notes": "",
            }
        )
    return rows


def collect_eval_rows(base_root: Path):
    rows = []
    if not base_root.exists():
        return rows

    for path in sorted(base_root.rglob("metrics.json")):
        path_str = str(path)
        if "eval_final" not in path_str:
            continue
        metrics = load_json(path)
        run_stats_path = path.parent / "run_stats.json"
        run_stats = load_json(run_stats_path) if run_stats_path.exists() else {}
        config = run_stats.get("config", {})
        rows.append(
            {
                "experiment_id": experiment_id_from_path(path, base_root),
                "family": "holdout_eval",
                "artifact_path": str(path),
                "model_path": config.get("model_path", ""),
                "map_location": config.get("map_location", ""),
                "holdout": config.get("gt_jsonl", ""),
                "micro_f1": safe_get(metrics, "micro", "f1"),
                "macro_f1": metrics.get("macro_f1"),
                "person_f1": safe_get(metrics, "per_label", "Person", "f1"),
                "location_f1": safe_get(metrics, "per_label", "Location", "f1"),
                "organization_f1": safe_get(metrics, "per_label", "Organization", "f1"),
                "runtime_hms": safe_get(run_stats, "summary", "runtime_hms", default=run_stats.get("runtime_hms")),
                "runtime_seconds": run_stats.get("runtime_seconds"),
                "status": "final",
                "notes": "",
            }
        )
    return rows


def collect_pseudolabelling_rows(pseudolabelling_root: Path):
    rows = []
    if not pseudolabelling_root.exists():
        return rows

    for path in sorted(pseudolabelling_root.rglob("cycle_summary.json")):
        cycle = load_json(path)
        run_dir = path.parent
        config = cycle.get("config", {})
        split_summary_path = run_dir / "05_split" / "summary.json"
        refit_stats_path = run_dir / "06_refit_stats.json"
        comparison_path = run_dir / "09_base_vs_refit_comparison.json"
        predictions_stats_path = run_dir / "01_predictions_stats.json"

        split_summary = load_json(split_summary_path) if split_summary_path.exists() else {}
        refit_stats = load_json(refit_stats_path) if refit_stats_path.exists() else {}
        comparison = load_json(comparison_path) if comparison_path.exists() else {}
        predictions_stats = load_json(predictions_stats_path) if predictions_stats_path.exists() else {}

        rows.append(
            {
                "experiment_id": experiment_id_from_path(path, pseudolabelling_root),
                "family": "pseudolabel_cycle",
                "artifact_path": str(path),
                "model_path": config.get("model_path", ""),
                "prediction_map_location": config.get("prediction_map_location", ""),
                "prediction_model_max_length": config.get("prediction_model_max_length", ""),
                "refit_mode": config.get("refit_mode", ""),
                "evaluate_refit": config.get("evaluate_refit", False),
                "runtime_hms": cycle.get("runtime_hms"),
                "runtime_seconds": cycle.get("runtime_seconds"),
                "prediction_runtime_hms": predictions_stats.get("runtime_hms"),
                "prediction_runtime_seconds": predictions_stats.get("runtime_seconds"),
                "entities_total": safe_get(predictions_stats, "summary", "total_entities"),
                "kept_count": safe_get(split_summary, "summary", "kept_count"),
                "discarded_count": safe_get(split_summary, "summary", "discarded_count"),
                "kept_score_mean": safe_get(split_summary, "summary", "kept_score_mean"),
                "train_size": safe_get(refit_stats, "data_summary", "train_size"),
                "pseudolabel_train_size": safe_get(refit_stats, "data_summary", "train_source_breakdown", "pseudolabel"),
                "base_micro_f1": safe_get(comparison, "micro_f1", "base"),
                "refit_micro_f1": safe_get(comparison, "micro_f1", "refit"),
                "micro_f1_delta": safe_get(comparison, "micro_f1", "delta"),
                "base_macro_f1": safe_get(comparison, "macro_f1", "base"),
                "refit_macro_f1": safe_get(comparison, "macro_f1", "refit"),
                "macro_f1_delta": safe_get(comparison, "macro_f1", "delta"),
                "status": "exploratory" if not config.get("evaluate_refit", False) else "final",
                "notes": "",
            }
        )
    return rows


def build_results_master(base_rows, eval_rows, pseudo_rows):
    master_rows = []
    for row in base_rows + eval_rows + pseudo_rows:
        master_rows.append(row)
    return master_rows


def build_baseline_table(eval_rows):
    preferred = []
    for row in eval_rows:
        experiment_id = row.get("experiment_id", "")
        model_path = row.get("model_path", "")
        if "gliner_multi_v21_raw" in experiment_id:
            label = "GLiNER cru (multi)"
        elif "multi_lr_wd_grid_bs16" in experiment_id:
            label = "Base finetunado (multi)"
        else:
            label = experiment_id
        preferred.append(
            {
                "model": label,
                "model_path": model_path,
                "micro_f1": row.get("micro_f1"),
                "macro_f1": row.get("macro_f1"),
                "person_f1": row.get("person_f1"),
                "location_f1": row.get("location_f1"),
                "organization_f1": row.get("organization_f1"),
                "holdout": row.get("holdout"),
            }
        )
    return preferred


def build_pseudolabel_table(pseudo_rows):
    return [
        {
            "experiment_id": row.get("experiment_id"),
            "refit_mode": row.get("refit_mode"),
            "prediction_map_location": row.get("prediction_map_location"),
            "prediction_runtime_hms": row.get("prediction_runtime_hms"),
            "entities_total": row.get("entities_total"),
            "kept_count": row.get("kept_count"),
            "kept_score_mean": row.get("kept_score_mean"),
            "pseudolabel_train_size": row.get("pseudolabel_train_size"),
            "base_micro_f1": row.get("base_micro_f1"),
            "refit_micro_f1": row.get("refit_micro_f1"),
            "micro_f1_delta": row.get("micro_f1_delta"),
            "base_macro_f1": row.get("base_macro_f1"),
            "refit_macro_f1": row.get("refit_macro_f1"),
            "macro_f1_delta": row.get("macro_f1_delta"),
        }
        for row in pseudo_rows
    ]


def build_runtime_table(eval_rows, pseudo_rows, base_rows):
    rows = []
    for row in eval_rows:
        rows.append(
            {
                "experiment": row.get("experiment_id"),
                "family": row.get("family"),
                "runtime_hms": row.get("runtime_hms"),
                "runtime_seconds": row.get("runtime_seconds"),
                "device": row.get("map_location", ""),
            }
        )
    for row in pseudo_rows:
        rows.append(
            {
                "experiment": row.get("experiment_id"),
                "family": row.get("family"),
                "runtime_hms": row.get("runtime_hms"),
                "runtime_seconds": row.get("runtime_seconds"),
                "device": row.get("prediction_map_location", ""),
            }
        )
    for row in base_rows:
        rows.append(
            {
                "experiment": row.get("experiment_id"),
                "family": row.get("family"),
                "runtime_hms": row.get("runtime_hms"),
                "runtime_seconds": row.get("runtime_seconds"),
                "device": "",
            }
        )
    return rows


def write_methodological_notes(path: Path):
    content = """# Methodological Notes

- Use `dd_corpus_small_test_final.json` as the external holdout.
- Treat results based on `dd_corpus_small_test_filtered.json` as legacy/exploratory unless explicitly revalidated.
- The principal supervised baseline is the fine-tuned `multi` model, not the raw backbone.
- Historical runtime measurements taken before explicit `map_location=cuda` support may reflect CPU inference rather than GPU inference.
- The base training pipeline originally dropped reports without annotated entities by default.
- Newer experiments may set `keep_empty_samples=true`; compare them explicitly against the earlier positive-only baseline.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main():
    args = parse_args()
    base_root = Path(args.base_root)
    pseudolabelling_root = Path(args.pseudolabelling_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = collect_nested_cv_rows(base_root)
    eval_rows = collect_eval_rows(base_root)
    pseudo_rows = collect_pseudolabelling_rows(pseudolabelling_root)
    master_rows = build_results_master(base_rows, eval_rows, pseudo_rows)

    master_fields = sorted({key for row in master_rows for key in row.keys()})
    baseline_rows = build_baseline_table(eval_rows)
    baseline_fields = [
        "model",
        "model_path",
        "micro_f1",
        "macro_f1",
        "person_f1",
        "location_f1",
        "organization_f1",
        "holdout",
    ]
    pseudolabel_rows = build_pseudolabel_table(pseudo_rows)
    pseudolabel_fields = [
        "experiment_id",
        "refit_mode",
        "prediction_map_location",
        "prediction_runtime_hms",
        "entities_total",
        "kept_count",
        "kept_score_mean",
        "pseudolabel_train_size",
        "base_micro_f1",
        "refit_micro_f1",
        "micro_f1_delta",
        "base_macro_f1",
        "refit_macro_f1",
        "macro_f1_delta",
    ]
    runtime_rows = build_runtime_table(eval_rows, pseudo_rows, base_rows)
    runtime_fields = ["experiment", "family", "runtime_hms", "runtime_seconds", "device"]

    write_csv(output_dir / "results_master.csv", master_rows, master_fields)
    write_csv(output_dir / "tables" / "table_baselines.csv", baseline_rows, baseline_fields)
    write_csv(output_dir / "tables" / "table_pseudolabel_probes.csv", pseudolabel_rows, pseudolabel_fields)
    write_csv(output_dir / "tables" / "table_runtime.csv", runtime_rows, runtime_fields)

    write_markdown_table(output_dir / "tables" / "table_baselines.md", baseline_rows, baseline_fields, "Baselines")
    write_markdown_table(
        output_dir / "tables" / "table_pseudolabel_probes.md",
        pseudolabel_rows,
        pseudolabel_fields,
        "Pseudolabel Probes",
    )
    write_markdown_table(output_dir / "tables" / "table_runtime.md", runtime_rows, runtime_fields, "Runtime Summary")
    write_methodological_notes(output_dir / "notes" / "methodological_notes.md")

    manifest = {
        "base_root": str(base_root),
        "pseudolabelling_root": str(pseudolabelling_root),
        "output_dir": str(output_dir),
        "rows": {
            "base_nested_cv": len(base_rows),
            "holdout_eval": len(eval_rows),
            "pseudolabel_cycle": len(pseudo_rows),
            "results_master": len(master_rows),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
