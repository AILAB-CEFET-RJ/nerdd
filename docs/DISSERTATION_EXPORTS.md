# Dissertation Exports

Use `src/tools/export_thesis_tables.py` to build a compact export package for dissertation writing without exposing the workstation.

## Goal

Generate:

- `results_master.csv`
- baseline tables
- pseudolabel probe tables
- runtime tables
- methodological notes

under:

- `artifacts/dissertation_exports/`

## Example

From `src/` on the workstation:

```bash
python3 tools/export_thesis_tables.py \
  --base-root ./artifacts/base_model_training \
  --pseudolabelling-root ./artifacts/pseudolabelling \
  --output-dir ./artifacts/dissertation_exports
```

## Main Outputs

- `artifacts/dissertation_exports/results_master.csv`
- `artifacts/dissertation_exports/tables/table_baselines.csv`
- `artifacts/dissertation_exports/tables/table_baselines.md`
- `artifacts/dissertation_exports/tables/table_pseudolabel_probes.csv`
- `artifacts/dissertation_exports/tables/table_pseudolabel_probes.md`
- `artifacts/dissertation_exports/tables/table_runtime.csv`
- `artifacts/dissertation_exports/tables/table_runtime.md`
- `artifacts/dissertation_exports/notes/methodological_notes.md`

## Recommended Workflow

1. Run the exporter after each major experiment completes.
2. Share only `artifacts/dissertation_exports/` with the dissertation writer.
3. Treat the exporter output as the canonical writing package.
