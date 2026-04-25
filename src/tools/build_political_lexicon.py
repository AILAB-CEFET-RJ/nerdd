#!/usr/bin/env python3
"""Command-line driver for political_lexicon_builder.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from political_lexicon_builder import (
    build_political_person_lexicon,
    read_municipalities_file,
    write_lexicon_csv,
)


def find_repo_root() -> Path:
    """Return repo root when this script lives in <repo>/src/tools."""
    here = Path(__file__).resolve()
    if here.parent.name == "tools" and here.parent.parent.name == "src":
        return here.parents[2]
    return Path.cwd()


REPO_ROOT = find_repo_root()


def parse_years(value: str) -> list[int]:
    years: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            years.append(int(part))
    return years


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a CSV lexicon of political-person names and ballot-name aliases "
            "from TSE open candidate data."
        )
    )
    parser.add_argument(
        "--municipios",
        nargs="*",
        default=None,
        help="Municipalities to include, e.g. Nilópolis Mesquita 'Nova Iguaçu'.",
    )
    parser.add_argument(
        "--municipios-file",
        default=None,
        help="UTF-8 text file with one municipality per line.",
    )
    parser.add_argument(
        "--no-municipio-filter",
        action="store_true",
        help=(
            "Do not filter by municipality. Useful for statewide offices such as "
            "DEPUTADO FEDERAL and DEPUTADO ESTADUAL."
        ),
    )
    parser.add_argument(
        "--anos",
        default="2024,2020",
        help=(
            "Comma-separated election years. Examples: 2024,2020,2016 for vereadores; "
            "2022,2018 for deputados."
        ),
    )
    parser.add_argument("--uf", default="RJ", help="State abbreviation. Default: RJ.")
    parser.add_argument(
        "--cargo",
        default="VEREADOR",
        help=(
            "Office(s) to filter. Accepts a comma-separated list, e.g. "
            "VEREADOR or DEPUTADO FEDERAL,DEPUTADO ESTADUAL. Default: VEREADOR."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "processed" / "lexico_politicos_locais.csv"),
        help="Output CSV path. Default: <repo>/data/processed/lexico_politicos_locais.csv.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "data" / "raw"),
        help="Directory to store downloaded TSE ZIP files. Default: <repo>/data/raw.",
    )
    parser.add_argument(
        "--elected-only",
        action="store_true",
        help="Keep only candidates whose totalization status indicates election.",
    )
    parser.add_argument(
        "--no-derived-aliases",
        action="store_true",
        help="Do not generate weaker derived aliases from ballot names.",
    )

    args = parser.parse_args()

    municipios: list[str] = []
    if args.municipios_file:
        municipios.extend(read_municipalities_file(args.municipios_file))
    if args.municipios:
        municipios.extend(args.municipios)

    cargos = parse_csv_list(args.cargo)

    if args.no_municipio_filter:
        municipios_filter = None
    else:
        # Sensible default for the motivating vereador example.
        if not municipios:
            municipios = [
                "Nilópolis",
                "Mesquita",
                "Nova Iguaçu",
                "Duque de Caxias",
                "São João de Meriti",
                "Belford Roxo",
                "Niterói",
                "São Gonçalo",
            ]
        municipios_filter = municipios

    rows = build_political_person_lexicon(
        municipios=municipios_filter,
        anos_eleicao=parse_years(args.anos),
        uf=args.uf,
        cargo=cargos,
        cache_dir=args.cache_dir,
        elected_only=args.elected_only,
        include_derived_aliases=not args.no_derived_aliases,
    )

    output = Path(args.output)
    write_lexicon_csv(rows, output)

    print(f"Wrote {len(rows)} rows to {output}")
    print("Municipality filter:", "OFF" if municipios_filter is None else ", ".join(municipios_filter))
    print("Cargo filter:", ", ".join(cargos))
    print("Years:", args.anos)
    print("Elected only:", args.elected_only)


if __name__ == "__main__":
    main()
