#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parte 7 — Treino (Fine-tuning) do GLiNER com GPU + Early Stopping
-----------------------------------------------------------------
- Lê um JSONL (ou pasta com `mantidos.jsonl`) gerado pela separação.
- Filtra registros válidos (texto não-vazio + ao menos 1 entidade).
- Usa GPU automaticamente (ou `--device cuda`).
- Treina por 10 épocas com Early Stopping (paciência=3) por padrão.
- Salva o modelo final em `--out_dir`.

Compatibilidade/robustez:
- Faz *monkey patch* em funções problemáticas do `treino.py` para evitar
  o erro `tokens, spans = []` e garantir tokenização por espaços com spans.

Uso (exemplo):
  python parte7_treino.py \
    --input ./separacao_treino \
    --out_dir ./gliner_finetuned \
    --base_model ../best_overall_gliner_model \
    --epochs 10 --patience 3 \
    --device cuda --workers 2
"""
from __future__ import annotations

import argparse
import json
import os
import random
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

# =============================
# Aceleração / GPU
# =============================
try:
    import torch
except Exception:
    torch = None  # fallback sem torch


def _select_device(user_choice: Optional[str] = None) -> str:
    if user_choice in {"cuda", "cpu"}:
        return user_choice
    if torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
            return "cuda"
    return "cpu"


def _configure_accel(device: str, enable_tf32: bool = True) -> None:
    if torch is None:
        return
    try:
        if device == "cuda" and getattr(torch.backends, "cuda", None) is not None:
            if enable_tf32:
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                except Exception:
                    pass
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    except Exception as e:
        print(f"[warn] Falha ao configurar aceleração: {e}")


# =============================
# Dependências locais do projeto
# =============================
try:
    from predicao import load_gliner
except Exception as e:
    raise RuntimeError(f"Falha ao importar load_gliner de predicao.py: {e}")

try:
    import treino as treino_mod
    from treino import treinar_gliner
except Exception as e:
    raise RuntimeError(f"Falha ao importar treino.py: {e}")


# =============================
# Monkey patch de funções frágeis do treino.py
# =============================
# Corrige _whitespace_tokenize_with_char_spans e, se existir, 
# garante comportamento safe em _char_to_token_spans.

def _safe_ws_tokenize_with_char_spans(text: str):
    """Tokeniza por espaços e retorna (tokens, spans), mesmo para string vazia."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    n = len(text)
    tokens, spans = [], []
    i = 0
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and not text[i].isspace():
            i += 1
        end = i
        tokens.append(text[start:end])
        spans.append((start, end))
    return tokens, spans


# Aplica o patch se essas funções existirem no módulo
if hasattr(treino_mod, "_whitespace_tokenize_with_char_spans"):
    treino_mod._whitespace_tokenize_with_char_spans = _safe_ws_tokenize_with_char_spans  # type: ignore


# =============================
# Utilitários de IO e preparo
# =============================
TEXT_KEYS = ("text", "relato", "texto", "descricao", "description")
CANDIDATE_MODELS = [
    "best_overall_gliner_model",
    "best_overall_gliner",
]

# rótulos permitidos para NER (pedido: apenas estes)
DEFAULT_ALLOWED_LABELS = {"Location", "Organization", "Person"}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    return items


def _ensure_text(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    for k in TEXT_KEYS:
        v = out.get(k)
        if isinstance(v, str) and v.strip():
            out["text"] = v.strip()
            return out
    out.setdefault("text", "")
    return out


def _filter_has_entities(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in items:
        ents = r.get("entities") or r.get("ner") or []
        txt = (r.get("text") or r.get("relato") or "").strip()
        if isinstance(ents, list) and len(ents) > 0 and len(txt) > 0:
            rr = dict(r)
            rr["text"] = txt
            out.append(rr)
    return out


def _filter_entities_by_label(items: List[Dict[str, Any]], allowed: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in items:
        ents = r.get("entities") if r.get("entities") is not None else r.get("ner")


def _pick_default_model(user_model: Optional[str]) -> str:
    if user_model:
        return user_model
    for cand in CANDIDATE_MODELS:
        if Path(cand).exists():
            return cand
    return CANDIDATE_MODELS[0]


def _split_train_val(items: List[Dict[str, Any]], val_ratio: float, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    nv = max(1, int(round(n * val_ratio)))
    val = items[:nv]
    train = items[nv:]
    if not train:
        train, val = items, []
    return train, val


# =============================
# Pipeline de treino
# =============================

def run(input_path: str | Path,
        out_dir: str | Path,
        base_model: Optional[str] = None,
        epochs: int = 10,
        patience: int = 3,
        val_jsonl: Optional[str] = None,
        val_ratio: float = 0.1,
        batch_size: int = 8,
        workers: int = 2,
        device: Optional[str] = None,
        use_compile: bool = True,
        tf32: bool = True,
        lr: float = 3e-5,
        weight_decay: float = 0.01,
        seed: int = 42) -> None:

    in_path = Path(input_path)
    if in_path.is_dir():
        cand = in_path / "preds002_iso_mantidos.jsonl"
        if not cand.exists():
            raise FileNotFoundError(f"Pasta de separação não contém mantidos.jsonl: {cand}")
        train_items = _read_jsonl(cand)
    else:
        train_items = _read_jsonl(in_path)

    train_items = _filter_has_entities(train_items)
    if not train_items:
        raise RuntimeError("Nenhum exemplo com texto e entidades no arquivo de treino.")

    # validação
    if val_jsonl:
        val_items = _filter_has_entities(_read_jsonl(Path(val_jsonl)))
    else:
        train_items, val_items = _split_train_val(train_items, val_ratio=val_ratio, seed=seed)

    model_name = _pick_default_model(base_model)
    model, _tok, _dev = load_gliner(model_name)

    # ==== GPU & aceleração ====
    dev = _select_device(device)
    _configure_accel(dev, enable_tf32=tf32)
    try:
        if torch is not None and hasattr(model, "to"):
            model = model.to(dev)
        if use_compile and torch is not None and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                print("[accel] torch.compile ativado")
            except Exception as e:
                print(f"[accel] torch.compile indisponível: {e}")
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
                print("[accel] gradient_checkpointing_enable ativado")
            except Exception:
                pass
    except Exception as e:
        print(f"[warn] Não foi possível mover/otimizar modelo: {e}")

    os.makedirs(out_dir, exist_ok=True)

    # Chama o treino do projeto (spanlabel) com early stopping e workers
    treinar_gliner(
        model=model,
        train_recs=train_items,
        val_recs=val_items,
        tokenizer=None,
        out_dir=str(out_dir),
        num_epochs=int(epochs),
        paciencia=int(patience),
        batch_size=int(batch_size),
        lr=float(lr),
        wd=float(weight_decay),
        dl_num_workers=int(workers),
    )
    print(f"[ok] Treino concluído. Modelo salvo em: {out_dir}")


# =============================
# CLI
# =============================

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parte 7 — Fine-tuning GLiNER (GPU + early stopping + patches de robustez)")
    p.add_argument("--input", required=True, help="Caminho do JSONL (ou pasta com mantidos.jsonl)")
    p.add_argument("--out_dir", required=True, help="Pasta de saída do modelo fine-tunado")
    p.add_argument("--base_model", default=None, help="Modelo base (pasta ou nome). Padrão: tenta best_overall_gliner*_ ")
    p.add_argument("--epochs", type=int, default=10, help="Número de épocas (padrão: 10)")
    p.add_argument("--patience", type=int, default=3, help="Paciência do early stopping (padrão: 3)")
    p.add_argument("--val_jsonl", default=None, help="Arquivo JSONL de validação (opcional)")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Proporção do treino reservada para validação (padrão: 0.1)")
    p.add_argument("--batch_size", type=int, default=8, help="Tamanho do batch (padrão: 8)")
    p.add_argument("--workers", type=int, default=2, help="Num. de workers DataLoader (padrão: 2)")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Força dispositivo (padrão: auto)")
    p.add_argument("--no_compile", action="store_true", help="Desativa torch.compile")
    p.add_argument("--no_tf32", action="store_true", help="Desativa TF32 (matmul/cudnn)")
    p.add_argument("--lr", type=float, default=3e-5, help="Learning rate (padrão: 3e-5)")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (padrão: 0.01)")
    p.add_argument("--seed", type=int, default=42, help="Seed para split/reprodutibilidade")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    run(
        input_path=args.input,
        out_dir=args.out_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        patience=args.patience,
        val_jsonl=args.val_jsonl,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        use_compile=(not args.no_compile),
        tf32=(not args.no_tf32),
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
