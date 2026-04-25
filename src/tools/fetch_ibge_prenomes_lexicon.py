#!/usr/bin/env python3
"""Fetch a first-name lexicon from the official IBGE Nomes API ranking endpoint."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path


IBGE_RANKING_URL = "https://servicodados.ibge.gov.br/api/v2/censos/nomes/ranking"


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch IBGE first-name ranking as JSONL.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path.")
    parser.add_argument("--localidade", default="BR", help="IBGE locality id. Use BR for Brazil.")
    parser.add_argument(
        "--decada",
        action="append",
        default=[],
        help="Decade filter accepted by IBGE API. Repeat to include several decades.",
    )
    parser.add_argument(
        "--include-common-decades",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also fetch common decade rankings from 1930 to 2010.",
    )
    parser.add_argument(
        "--sexo",
        action="append",
        choices=("M", "F"),
        default=["M", "F"],
        help="Sex filter to fetch. Repeat to include both. Defaults to M and F.",
    )
    return parser.parse_args()


def fetch_ranking(*, localidade: str, sexo: str, decada: str = "") -> list[dict]:
    params_payload = {"localidade": localidade, "sexo": sexo}
    if decada:
        params_payload["decada"] = decada
    params = urllib.parse.urlencode(params_payload)
    url = f"{IBGE_RANKING_URL}?{params}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload:
        return []
    rows = payload[0].get("res", [])
    for row in rows:
        row["sexo"] = sexo
        row["localidade"] = localidade
        if decada:
            row["decada"] = decada
        row["source"] = "IBGE API Nomes ranking"
    return rows


def main():
    args = parse_args()
    merged: dict[str, dict] = {}
    decades = list(args.decada)
    if args.include_common_decades:
        decades.extend(["1930", "1940", "1950", "1960", "1970", "1980", "1990", "2000", "2010"])
    queries = [""] + sorted(set(decades))
    for sexo in args.sexo:
        for decada in queries:
            for row in fetch_ranking(localidade=args.localidade, sexo=sexo, decada=decada):
                name = str(row.get("nome", "")).strip().upper()
                if not name:
                    continue
                current = merged.setdefault(
                    name,
                    {
                        "nome": name,
                        "frequencia": 0,
                        "ranking_min": row.get("ranking"),
                        "sexos": [],
                        "decadas": [],
                        "localidade": args.localidade,
                        "source": "IBGE API Nomes ranking",
                    },
                )
                current["frequencia"] += int(row.get("frequencia") or 0)
                if sexo not in current["sexos"]:
                    current["sexos"].append(sexo)
                if decada and decada not in current["decadas"]:
                    current["decadas"].append(decada)
                if row.get("ranking") is not None:
                    current["ranking_min"] = min(int(current["ranking_min"]), int(row["ranking"]))

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(merged.values(), key=lambda item: (int(item["ranking_min"]), item["nome"]))
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {len(rows)} names to {output_path}")


if __name__ == "__main__":
    main()
