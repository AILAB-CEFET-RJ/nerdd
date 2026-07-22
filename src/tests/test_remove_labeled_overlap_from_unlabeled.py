import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.remove_labeled_overlap_from_unlabeled import (
    build_labeled_key_index,
    filter_unlabeled_pool,
    parse_strategy_list,
)


class RemoveLabeledOverlapFromUnlabeledTests(unittest.TestCase):
    def test_removes_overlap_with_punctuation_and_accent_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labeled_path = tmp_path / "labeled.json"
            input_path = tmp_path / "pool.jsonl"
            output_path = tmp_path / "filtered.jsonl"
            removed_path = tmp_path / "removed.jsonl"

            labeled_path.write_text(
                json.dumps(
                    [
                        {
                            "text": "Denuncia na Rua Sao Joao , proximo ao mercado .",
                            "spans": [],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "assunto": "Roubos em Geral",
                                "relato": "Denúncia na Rua São João, próximo ao mercado.",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "assunto": "Armas",
                                "relato": "Outro relato sem interseção.",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            strategies = parse_strategy_list("exact,whitespace_lower,alnum_lower,ascii_alnum_lower")
            key_index, _ = build_labeled_key_index(
                [str(labeled_path)],
                labeled_text_field="auto",
                strategies=strategies,
            )
            summary = filter_unlabeled_pool(
                input_jsonl=str(input_path),
                output_jsonl=str(output_path),
                removed_jsonl=str(removed_path),
                text_field="auto",
                key_index=key_index,
                strategies=strategies,
                preview_limit=10,
            )

            kept = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            removed = [json.loads(line) for line in removed_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary["rows_in"], 2)
            self.assertEqual(summary["rows_kept"], 1)
            self.assertEqual(summary["rows_removed"], 1)
            self.assertEqual(summary["removed_by_strategy"], {"ascii_alnum_lower": 1})
            self.assertEqual(kept[0]["assunto"], "Armas")
            self.assertEqual(removed[0]["assunto"], "Roubos em Geral")


if __name__ == "__main__":
    unittest.main()
