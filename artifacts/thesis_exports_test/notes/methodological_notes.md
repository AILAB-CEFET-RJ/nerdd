# Methodological Notes

- Use `dd_corpus_small_test_final.json` as the external holdout.
- Treat results based on `dd_corpus_small_test_filtered.json` as legacy/exploratory unless explicitly revalidated.
- The principal supervised baseline is the fine-tuned `multi` model, not the raw backbone.
- Historical runtime measurements taken before explicit `map_location=cuda` support may reflect CPU inference rather than GPU inference.
- The base training pipeline originally dropped reports without annotated entities by default.
- Newer experiments may set `keep_empty_samples=true`; compare them explicitly against the earlier positive-only baseline.
