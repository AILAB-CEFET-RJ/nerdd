#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parte 9 — Limpar arquivos *_descartados.jsonl e salvar apenas campos desejados
-----------------------------------------------------------------------------
Lê os arquivos:
  - preds001_ts_descartados.jsonl  ->  preds_002_ts.jsonl
  - preds001_iso_descartados.jsonl ->  preds_002_iso.jsonl
  - preds001_raw_descartados.jsonl ->  preds_002_raw.jsonl

De cada registro, mantém SOMENTE os campos:
  assunto, relato, logradouroLocal, bairroLocal, cidadeLocal, pontodeReferenciaLocal

Uso:
  python parte9_limpa_descartados.py \
    --in_dir  ./ \
    --out_dir ./

Observações:
- Aceita JSONL (um JSON por linha) ou um JSON único/lista (nós gravamos sempre JSONL).
- Se algum campo estiver ausente no registro, ele é criado como string vazia.
- Pula linhas vazias e avisa se o arquivo de entrada não existir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

FIELDS = [
    "assunto",
    "relato",
    "logradouroLocal",
    "bairroLocal",
    "cidadeLocal",
    "pontodeReferenciaLocal",
]

MAP = {
    "preds002_ts_descartados.jsonl":  "raw_003_ts.jsonl",
    "preds002_iso_descartados.jsonl": "raw_003_iso.jsonl",
    "preds002_raw_descartados.jsonl": "raw_003_raw.jsonl",
}


def _iter_json_any(path: Path) -> Iterator[Dict]:
    """Itera registros a partir de um JSONL (preferência) ou JSON (objeto/lista)."""
    text = path.read_text(encoding="utf-8")
    # tenta JSON direto
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # trata como JSONL
        for i, ln in enumerate(text.splitlines(), start=1):
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}: linha {i} não é JSON válido: {e}")
        return
    # JSON comum
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        yield obj
    else:
        raise ValueError(f"{path.name}: JSON inválido (esperado objeto ou lista)")


def _project(rec: Dict) -> Dict:
    return {k: (rec.get(k, "") if isinstance(rec.get(k, None), (str, int, float)) or rec.get(k, None) is None else rec.get(k, "")) for k in FIELDS}


def _write_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def process(in_dir: Path, out_dir: Path) -> None:
    for src_name, dst_name in MAP.items():
        src = in_dir / src_name
        dst = out_dir / dst_name
        if not src.exists():
            print(f"[aviso] arquivo não encontrado: {src}")
            continue
        out_rows: List[Dict] = []
        for rec in _iter_json_any(src):
            out_rows.append(_project(rec))
        _write_jsonl(out_rows, dst)
        print(f"[ok] {src.name} -> {dst}  (n={len(out_rows)})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Limpa *_descartados.jsonl, mantendo só os campos pedidos")
    ap.add_argument("--in_dir", default=".", help="Pasta de entrada (onde estão os preds001_*_descartados.jsonl)")
    ap.add_argument("--out_dir", default=".", help="Pasta de saída para preds_002_*.jsonl")
    args = ap.parse_args()
    process(Path(args.in_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
