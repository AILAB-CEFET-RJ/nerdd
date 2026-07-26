#!/usr/bin/env python3
"""Amplia spans Location para incluir designadores geográficos anteriores.

O texto das entradas não é alterado. Para cada span corrigido, somente o
offset ``start`` é deslocado para o início do designador detectado.

python scripts/fix_location_boundaries.py \
  ./data/dd_corpus_small_train.json \
  ./data/dd_corpus_small_train_fixed.json \
  --log ./data/location_fixes_train.log \
  --validate

python scripts/fix_location_boundaries.py \
  ./data/dd_corpus_small_test.json \
  ./data/dd_corpus_small_test_fixed.json \
  --log ./data/location_fixes_test.log \
  --validate

"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_DESIGNATORS = (
    "rua",
    "avenida",
    "av.",
    "av",
    "bairro",
    "favela",
    "comunidade",
    "morro",
    "complexo",
    "estrada",
    "travessa",
    "praça",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Amplia spans Location para incorporar um designador geográfico "
            "imediatamente anterior."
        )
    )
    parser.add_argument("input", type=Path, help="Corpus JSON de entrada.")
    parser.add_argument("output", type=Path, help="Corpus JSON de saída.")
    parser.add_argument(
        "--log",
        type=Path,
        help="Arquivo de log (padrão: <arquivo-de-saída>.log).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detecta e registra alterações, mas não grava o corpus de saída.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Valida estrutura, offsets, texto dos spans e sobreposições ao final.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentação do JSON de saída (padrão: 2; use 0 para JSON compacto).",
    )
    parser.add_argument(
        "--lookbehind",
        type=int,
        default=80,
        help="Número máximo de caracteres examinados antes do span (padrão: 80).",
    )
    parser.add_argument(
        "--designators",
        nargs="+",
        default=list(DEFAULT_DESIGNATORS),
        metavar="TERMO",
        help="Lista fechada de designadores aceitos.",
    )
    return parser.parse_args()


def make_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fix_location_boundaries")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def build_pattern(designators: list[str]) -> re.Pattern[str]:
    # Termos maiores primeiro evita que "av" capture o prefixo de "av.".
    alternatives = "|".join(
        re.escape(item) for item in sorted(set(designators), key=len, reverse=True)
    )
    return re.compile(
        rf"(?i)(?<!\w)(?:{alternatives})\s+(?:d[oa]s?\s+|de\s+)?$"
    )


def load_corpus(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError("A raiz do corpus deve ser um array JSON.")
    return data


def overlaps_other_span(
    spans: list[dict[str, Any]], current_index: int, new_start: int, end: int
) -> bool:
    for index, other in enumerate(spans):
        if index == current_index:
            continue
        other_start = other.get("start")
        other_end = other.get("end")
        if not isinstance(other_start, int) or not isinstance(other_end, int):
            continue
        if new_start < other_end and other_start < end:
            return True
    return False


def validate_corpus(data: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row_index, item in enumerate(data):
        text = item.get("text")
        spans = item.get("spans", [])
        if not isinstance(text, str):
            errors.append(f"entrada {row_index}: campo 'text' ausente ou não textual")
            continue
        if not isinstance(spans, list):
            errors.append(f"entrada {row_index}: campo 'spans' não é uma lista")
            continue

        intervals: list[tuple[int, int, int]] = []
        for span_index, span in enumerate(spans):
            start, end = span.get("start"), span.get("end")
            label = span.get("label")
            if not isinstance(start, int) or not isinstance(end, int):
                errors.append(
                    f"entrada {row_index}, span {span_index}: offsets não inteiros"
                )
                continue
            if not (0 <= start < end <= len(text)):
                errors.append(
                    f"entrada {row_index}, span {span_index}: "
                    f"offsets inválidos [{start}, {end})"
                )
            if not isinstance(label, str) or not label:
                errors.append(
                    f"entrada {row_index}, span {span_index}: label inválido"
                )
            intervals.append((start, end, span_index))

        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                errors.append(
                    f"entrada {row_index}: sobreposição entre spans "
                    f"{previous[2]} e {current[2]}"
                )
    return errors


def apply_fixes(
    data: list[dict[str, Any]],
    pattern: re.Pattern[str],
    lookbehind: int,
    logger: logging.Logger,
) -> tuple[int, int]:
    changed = 0
    skipped_overlap = 0

    for row_index, item in enumerate(data):
        text = item.get("text")
        spans = item.get("spans")
        if not isinstance(text, str) or not isinstance(spans, list):
            continue

        for span_index, span in enumerate(spans):
            if span.get("label") != "Location":
                continue
            start, end = span.get("start"), span.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if not (0 <= start < end <= len(text)):
                continue

            window_start = max(0, start - lookbehind)
            prefix = text[window_start:start]
            match = pattern.search(prefix)
            if not match:
                continue

            new_start = window_start + match.start()
            if overlaps_other_span(spans, span_index, new_start, end):
                skipped_overlap += 1
                logger.warning(
                    "SKIP_OVERLAP | entry=%d | span=%d | old_offsets=[%d,%d) "
                    "| proposed_offsets=[%d,%d) | old=%r | proposed=%r",
                    row_index,
                    span_index,
                    start,
                    end,
                    new_start,
                    end,
                    text[start:end],
                    text[new_start:end],
                )
                continue

            old_value = text[start:end]
            new_value = text[new_start:end]
            span["start"] = new_start
            changed += 1
            logger.info(
                "CHANGE | entry=%d | span=%d | old_offsets=[%d,%d) "
                "| new_offsets=[%d,%d) | old=%r | new=%r",
                row_index,
                span_index,
                start,
                end,
                new_start,
                end,
                old_value,
                new_value,
            )
    return changed, skipped_overlap


def main() -> int:
    args = parse_args()
    log_path = args.log or args.output.with_suffix(args.output.suffix + ".log")
    logger = make_logger(log_path)

    try:
        if args.lookbehind < 1:
            raise ValueError("--lookbehind deve ser maior que zero.")
        if not args.designators:
            raise ValueError("--designators requer ao menos um termo.")
        if not args.dry_run and args.input.resolve() == args.output.resolve():
            raise ValueError(
                "Entrada e saída devem ser arquivos diferentes para preservar o original."
            )

        source = load_corpus(args.input)
        result = copy.deepcopy(source)
        pattern = build_pattern(args.designators)

        logger.info(
            "START | input=%s | output=%s | dry_run=%s | entries=%d",
            args.input,
            args.output,
            args.dry_run,
            len(source),
        )
        changed, skipped = apply_fixes(
            result, pattern, args.lookbehind, logger
        )

        if args.validate:
            errors = validate_corpus(result)
            if errors:
                for error in errors:
                    logger.error("VALIDATION | %s", error)
                raise ValueError(
                    f"Validação falhou com {len(errors)} erro(s); saída não gravada."
                )
            logger.info("VALIDATION_OK | entries=%d", len(result))

        if args.dry_run:
            logger.info(
                "DONE | changes=%d | skipped_overlap=%d | output_written=false",
                changed,
                skipped,
            )
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as stream:
                json.dump(
                    result,
                    stream,
                    ensure_ascii=False,
                    indent=args.indent or None,
                )
                stream.write("\n")
            logger.info(
                "DONE | changes=%d | skipped_overlap=%d | output_written=true",
                changed,
                skipped,
            )
        return 0
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.error("ABORTED | %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
