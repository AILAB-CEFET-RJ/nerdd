#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parte 1 — Predição
------------------
Executa a predição do modelo GLiNER já "marcado" no código/projeto
(pasta do modelo local) e salva os registros com entidades em um arquivo
separado (JSONL). Não faz calibração ou seleção iterativa — apenas
carrega o modelo e prediz.

Uso:
    python parte1_predicao.py \
        --input /caminho/para/dados.json[l] \
        --out   ./saida/predicoes.jsonl \
        [--model /caminho/para/best_overall_gliner_model]

Observações:
- O script tenta escolher automaticamente entre as pastas
  "best_overall_gliner_model" e "best_overall_gliner" se --model
  não for informado.
- O arquivo de entrada pode ser JSON (lista de registros ou objeto com
  chave data/records/items) ou JSONL (um JSON por linha).
- Campos possíveis de texto: "text", "relato", "texto", "descricao", "description".
- Saída: JSONL, mantendo os campos originais + campo padronizado "text"
  e a lista "entities" (cada uma com start, end, label e score quando
  fornecido pelo modelo).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Torch para controle explícito de GPU/precision
try:
    import torch
except Exception:
    torch = None  # mantém funcional sem torch (ex.: ambiente sem GPU)

# ---- Dependências locais ----------------------------------------------------
try:
    from predicao import load_gliner, predict_record
except Exception as e:
    raise RuntimeError("Falha ao importar utilidades de predição (predicao.py): %s" % e)

# ----------------------------------------------------------------------------
TEXT_KEYS = ("text", "relato", "texto", "descricao", "description")
CANDIDATE_MODELS = [
    "best_overall_gliner_model",  # usado em main_iterativo_ts.py
    "best_overall_gliner",        # usado em main_iterativo.py
]

# ----------------------------------------------------------------------------
# IO helpers

def _parse_jsonl(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, ln in enumerate(text.splitlines(), start=1):
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError as e:
            raise ValueError(f"Entrada parece JSONL, mas a linha {i} não é JSON válido: {e}")
    return out


def read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    text = p.read_text(encoding="utf-8")
    # Tenta JSON comum primeiro
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Trata como JSONL
        return _parse_jsonl(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("data", "records", "items", "itens", "registros"):
            v = obj.get(k)
            if isinstance(v, list):
                return list(v)
        # Caso seja um único registro
        return [obj]
    raise ValueError("Formato JSON inválido.")


def write_jsonl(items: Iterable[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# Normalização de texto

def _get_text(r: Dict[str, Any]) -> str:
    for k in TEXT_KEYS:
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _ensure_text_key(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Garante rec["text"] preenchido a partir de chaves alternativas."""
    out = dict(rec)
    txt = _get_text(out)
    out["text"] = txt
    return out


# ----------------------------------------------------------------------------
# Escolha do modelo

def _pick_default_model(user_model: Optional[str]) -> str:
    if user_model:
        return user_model
    # Procura candidatos em ordem
    for cand in CANDIDATE_MODELS:
        if Path(cand).exists():
            return cand
    # Se nada existir localmente, retorna o primeiro nome (pasta esperada)
    return CANDIDATE_MODELS[0]


# ----------------------------------------------------------------------------
# Predição

def _select_device(user_device: Optional[str]) -> str:
    if user_device:
        return user_device
    if torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _map_dtype(user_dtype: Optional[str]):
    if not user_dtype:
        return None
    user_dtype = user_dtype.lower()
    if user_dtype in {"fp16", "float16", "half"}:
        return getattr(torch, "float16", None) if torch else None
    if user_dtype in {"bf16", "bfloat16"}:
        return getattr(torch, "bfloat16", None) if torch else None
    if user_dtype in {"fp32", "float32", "32"}:
        return getattr(torch, "float32", None) if torch else None
    return None


def predict_file(input_path: str | Path, out_path: str | Path, model_name: Optional[str] = None,
                 device: Optional[str] = None, dtype: Optional[str] = None) -> None:
    # 1) Carrega dados
    registros = read_json_or_jsonl(input_path)

    # 2) Escolhe modelo
    model_to_use = _pick_default_model(model_name)

    # 3) Dispositivo e precisão
    chosen_device = _select_device(device)
    chosen_dtype = _map_dtype(dtype)

    # 4) Carrega modelo/tokenizer
    model, tokenizer, loaded_device = load_gliner(model_to_use)

    # 4.1) Move para o dispositivo desejado, se necessário
    try:
        if torch is not None and hasattr(model, "to"):
            model = model.to(chosen_device)
            # Ajuste de dtype opcional (apenas em GPU)
            if chosen_dtype is not None and str(chosen_device).startswith("cuda"):
                if chosen_dtype == torch.float16 and hasattr(model, "half"):
                    model = model.half()
                elif chosen_dtype == torch.bfloat16 and hasattr(model, "bfloat16"):
                    model = model.bfloat16()
                # fp32 é o padrão; não precisa alterar
        if torch is not None and hasattr(torch, "set_grad_enabled"):
            torch.set_grad_enabled(False)
            # Afinar desempenho quando disponível
            if hasattr(torch, "set_float32_matmul_precision"):
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
            if getattr(torch.backends, "cuda", None) is not None:
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                except Exception:
                    pass
    except Exception as e:
        print(f"[aviso] Não foi possível configurar GPU/precision: {e}")

    print(f"[info] Modelo: {model_to_use} | device={chosen_device}{' | dtype='+str(chosen_dtype) if chosen_dtype else ''}")

    # 5) Predição registro a registro
    saida: List[Dict[str, Any]] = []
    for idx, r in enumerate(registros, start=1):
        rec = _ensure_text_key(r)
        try:
            rec_pred = predict_record(model, tokenizer, rec)
        except Exception as e:
            # Mantém o registro, mesmo em caso de erro, para auditoria
            rec_pred = dict(rec)
            rec_pred.setdefault("entities", [])
            rec_pred["erro_predicao"] = str(e)
        saida.append(rec_pred)
        if idx % 1000 == 0:
            print(f"[predicao] processados {idx} registros...")

    # 6) Salva em arquivo separado (JSONL)
    write_jsonl(saida, out_path)
    print(f"[ok] Predições salvas em: {out_path}  (n={len(saida)})")


# ----------------------------------------------------------------------------
# CLI

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parte 1 — Predição com GLiNER (sem calibração)")
    p.add_argument("--input", required=True, help="Caminho do JSON/JSONL de entrada")
    p.add_argument("--out", required=True, help="Caminho do JSONL de saída (predições)")
    p.add_argument("--model", default=None, help="Pasta/nome do modelo (opcional). Se ausente, tenta candidatos locais.")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"], help="Força o dispositivo (cuda|cpu). Padrão: auto.")
    p.add_argument("--dtype", default=None, choices=["fp16", "bf16", "fp32"], help="Precisão do modelo (apenas CUDA).")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    predict_file(args.input, args.out, model_name=args.model, device=args.device, dtype=args.dtype)


if __name__ == "__main__":
    main()
