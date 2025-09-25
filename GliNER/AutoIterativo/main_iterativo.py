# main_iterativo.py (re-escrito)
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- Calibração: somente Isotonic Regression (global) ---
from calibracao_iso import (
    fit_isotonic_from_csv,
    add_score_iso_to_record,
)

# --- Predição/utilidades existentes ---
from predicao import (
    load_gliner,
    predict_record,
    apply_metadata_adjustments,
    compute_score_relato,
    add_score_relato_por_entidade,
)

# ---------------- Configs padrão ----------------
CSV_CALIBRACAO = "/mnt/localssd/comparacao_calibracao.csv"
ITERACOES = 5
TAU_SEQ = [0.90, 0.80, 0.65, 0.50, 0.35]
TAU = 0.80  # fallback quando TAU_SEQ acabar
ALPHA = 1.2
MODEL_BASE = "best_overall_gliner"
SAIDA_DIR = "saida_iterativa"

# --------------- IO helpers ---------------

def _read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Lê JSON **ou** JSONL automaticamente.
    - .jsonl/.ndjson => linha a linha
    - .json => tenta json.load; se der Extra data, cai para JSONL
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    # Tratamento direto por extensão
    if p.suffix.lower() in {".jsonl", ".ndjson"}:
        items: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8-sig") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Falha ao parsear JSONL na linha {ln}: {e}")
        return items

    # .json (ou extensão genérica): tenta json.load; se falhar por Extra data => JSONL
    with p.open("r", encoding="utf-8-sig") as f:
        # olhadinha para decidir rápido quando é um array JSON
        head = f.read(1024)
        f.seek(0)
        if head.lstrip().startswith("["):
            obj = json.load(f)
            if isinstance(obj, list):
                return obj  # lista de registros
            if isinstance(obj, dict):
                for key in ("data", "records", "amostras", "itens"):
                    if key in obj and isinstance(obj[key], list):
                        return obj[key]  # type: ignore[return-value]
                raise ValueError("JSON não está em formato esperado (lista ou {data:[...]})")
        try:
            obj = json.load(f)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                for key in ("data", "records", "amostras", "itens"):
                    if key in obj and isinstance(obj[key], list):
                        return obj[key]  # type: ignore[return-value]
                raise ValueError("JSON não está em formato esperado (lista ou {data:[...]})")
        except json.JSONDecodeError as e:
            # fallback: tratar como JSONL mesmo que a extensão seja .json
            if "Extra data" not in str(e):
                raise
            items: List[Dict[str, Any]] = []
            f.seek(0)
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e2:
                    raise ValueError(f"Falha ao parsear JSONL (arquivo com múltiplos objetos) na linha {ln}: {e2}")
            return items


def _write_jsonl(items: List[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


# --------------- Iteração principal ---------------

def iteracao(
    i: int,
    *,
    model,
    tokenizer,
    device,
    dados: List[Dict[str, Any]],
    iso_model,
    alpha: float,
    tau_iter: float,
    saida_dir: str | Path,
) -> Tuple[Any, Any, List[Dict[str, Any]]]:
    out_registros: List[Dict[str, Any]] = []
    descartados: List[Dict[str, Any]] = []

    for rec in dados:
        # 1) Predição de entidades
        rec_pred = predict_record(model, tokenizer, rec)

        # 2) Calibração (ISO) nos scores de entidade
        rec_pred = add_score_iso_to_record(rec_pred, iso_model=iso_model, score_in="score", score_out="score_ts")

        # 3) Ajustes por metadados (mantém a API, mesmo que não altere nada)
        rec_pred, _ = apply_metadata_adjustments(rec_pred, alpha=float(alpha))

        # 4) Score do relato (média dos scores calibrados por entidade)
        s_rel = compute_score_relato(rec_pred)
        if s_rel is not None:
            rec_pred["score_relato"] = float(s_rel)

        # também preenche score_relato por entidade (compat)
        rec_pred = add_score_relato_por_entidade(rec_pred)

        # 5) Filtragem por threshold
        if (s_rel is None) or (s_rel < tau_iter):
            descartados.append(rec_pred)
        else:
            out_registros.append(rec_pred)

    # 6) Persistência da iteração
    it_dir = Path(saida_dir) / f"iter_{i:02d}"
    _write_jsonl(out_registros, it_dir / "mantidos.jsonl")
    _write_jsonl(descartados, it_dir / "descartados.jsonl")
    print(f"[Iter {i}] mantidos={len(out_registros)} descartados={len(descartados)} (tau={tau_iter:.3f})")


    return model, tokenizer, descartados


# --------------- Função principal ---------------

def main(
    *,
    input_file: str | Path,
    csv_calibracao: str | Path = CSV_CALIBRACAO,
    iteracoes: int = ITERACOES,
    tau: float = TAU,
    alpha: float = ALPHA,
    model_base: str = MODEL_BASE,
    saida_dir: str | Path = SAIDA_DIR,
) -> None:
    Path(saida_dir).mkdir(parents=True, exist_ok=True)

    # 0) Calibração ISO (global)
    iso_model = None
    try:
        iso_model = fit_isotonic_from_csv(csv_calibracao, save_plots_to=saida_dir, plots_prefix="iso")
    except Exception as e:
        iso_model = None
        print(f"[warn] Não foi possível ajustar Isotonic Regression a partir de '{csv_calibracao}': {e}. Usando identidade (sem calibração).")
    else:
        print(f"[Main] Isotonic Regression ajustada a partir de '{csv_calibracao}'.")

    # 1) Carregar modelo/tokenizer
    model, tokenizer, device = load_gliner(model_base)

    # 2) Carregar dados
    dados = _read_json_or_jsonl(input_file)

    # 3) Loop iterativo
    current = list(dados)
    for i in range(1, int(iteracoes) + 1):
        tau_iter = TAU_SEQ[i - 1] if (i - 1) < len(TAU_SEQ) else float(tau)
        print(f"\n[Iter {i}] tau_iter={tau_iter:.3f} | calib={'ISO' if iso_model is not None else 'IDENT'} | alpha={alpha}")

        model, tokenizer, descartados = iteracao(
            i=i,
            model=model,
            tokenizer=tokenizer,
            device=device,
            dados=current,
            iso_model=iso_model,
            alpha=alpha,
            tau_iter=tau_iter,
            saida_dir=saida_dir,
        )

        # regra simples: próxima iteração foca nos descartados
        current = descartados if descartados else []
        if not current:
            print(f"[Iter {i}] Sem descartados restantes — encerrando.")
            break

    print("[Main] Concluído.")


# --------------- CLI ---------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pipeline iterativo com Isotonic Regression (global)")
    p.add_argument("--input", required=True, help="Caminho do JSON/JSONL de entrada")
    p.add_argument("--csv", default=CSV_CALIBRACAO, help="CSV para calibração (colunas Score/Validacao)")
    p.add_argument("--iter", type=int, default=ITERACOES, help="Número máximo de iterações")
    p.add_argument("--tau", type=float, default=TAU, help="Threshold padrão de score do relato")
    p.add_argument("--alpha", type=float, default=ALPHA, help="Fator de ajuste por metadados")
    p.add_argument("--model", default=MODEL_BASE, help="Modelo/base GLiNER (pasta ou nome HF)")
    p.add_argument("--out", default=SAIDA_DIR, help="Diretório de saída")
    args = p.parse_args()

    main(
        input_file=args.input,
        csv_calibracao=args.csv,
        iteracoes=args.iter,
        tau=args.tau,
        alpha=args.alpha,
        model_base=args.model,
        saida_dir=args.out,
    )
