# main_iterativo_ts.py — pipeline iterativo com Isotonic Regression (ISO)
# ----------------------------------------------------------------------------
# Funcionalidades principais
#  1) Lê uma base JSON/JSONL com registros contendo, no mínimo, um campo de texto
#     ("relato" ou "texto").
#  2) Executa predição com GLiNER por iteração (via predicao.py), aplica
#     Isotonic Regression global às entidades, calcula um score do relato
#     (média dos scores calibrados das entidades) e filtra por threshold (tau).
#  3) Garante um piso de seleção: pelo menos MIN_KEEP_FRAC (ex.: 10%) do conjunto
#     de entrada da iteração é mantido, rebaixando o threshold efetivo se
#     necessário (top-k por score_relato).
#  4) Salva, por iteração, dois arquivos JSONL: mantidos.jsonl e descartados.jsonl.
#  5) Nas iterações seguintes, processa apenas os descartados da iteração anterior,
#     a não ser que --continue_all seja usado (reprocessa todos em cada passo).
# ----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Evita barulho de warnings do HF/Transformers
warnings.filterwarnings("ignore")

# --- Dependências locais ------------------------------------------------------
# predicao.py deve fornecer as funções abaixo (interfaces mínimas usadas aqui):
#   load_gliner(model_name_or_path) -> (model, tokenizer)
#   predict_record(model, tokenizer, sample, labels=None) -> record_com_entities
#   compute_score_relato(record) -> float|None
#   add_score_relato_por_entidade(record) -> record
try:
    from predicao import (
        load_gliner,
        predict_record,
        compute_score_relato,
        add_score_relato_por_entidade,
    )
except Exception as e:
    raise RuntimeError(
        "Falha ao importar funções de 'predicao.py'. Verifique se o arquivo está no PYTHONPATH."
    ) from e

# calibracao_iso.py deve fornecer:
#   fit_isotonic_from_csv(csv_path, ...) -> IsotonicRegression
#   add_score_iso_to_record(record, iso_model, score_in='score', score_out='score_ts') -> record
try:
    from calibracao_iso import (
        fit_isotonic_from_csv,
        add_score_iso_to_record,
    )
except Exception as e:
    raise RuntimeError(
        "Falha ao importar funções de 'calibracao_iso.py'. Verifique se o arquivo está no PYTHONPATH."
    ) from e

# ----------------------------------------------------------------------------
# Parâmetros padrão
CSV_CALIBRACAO = "/mnt/data/comparacao_calibracao.csv"
ITERACOES = 5
TAU_SEQ_DEFAULT = [0.90, 0.80, 0.65, 0.50, 0.35]
TAU_FALLBACK = 0.80  # usado quando TAU_SEQ acabar
ALPHA = 1.2
MODEL_BASE = "best_overall_gliner_model"
SAIDA_DIR = "saida_iterativa_ts"
MIN_KEEP_FRAC_DEFAULT = 0.10  # mínimo de 10% mantidos por iteração


# --------------------------------------------------------------------------
# Utilitários de IO

def _read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Lê JSON *ou* JSONL automaticamente, independentemente da extensão.
    - .jsonl/.ndjson => um JSON por linha
    - .json => lista de objetos ou objeto único; se falhar, tenta auto-detectar JSONL
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    def _parse_jsonl_lines(txt: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, line in enumerate(txt.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Linha {i} não é um JSON válido: {line[:120]}... ({e})")
        return out

    # Detecta JSONL por heurística: linhas múltiplas e a primeira começa com { e contém }
    if "\n" in text and text.lstrip().startswith("{") and text.rstrip().endswith("}"):
        try:
            # Tenta como JSON (lista ou objeto único)
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
            elif isinstance(obj, dict):
                return [obj]
        except Exception:
            # Se falhar, tenta como JSONL
            return _parse_jsonl_lines(text)
    else:
        # Uma linha só: tenta JSON; se falhar, tenta JSONL
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, list) else [obj]
        except Exception:
            return _parse_jsonl_lines(text)


def _write_jsonl(path: str | Path, items: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Filtragem por tau + piso mínimo

@dataclass
class IterationConfig:
    i: int
    tau: float
    min_keep_frac: float
    alpha: float = ALPHA
    continue_all: bool = False


def _filtra_por_tau(registros: List[Dict[str, Any]], tau: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    mantidos: List[Dict[str, Any]] = []
    descartados: List[Dict[str, Any]] = []
    for r in registros:
        s = float(r.get("score_relato") or 0.0)
        (mantidos if s >= tau else descartados).append(r)
    return mantidos, descartados


def _enforce_min_keep_fraction(
    mantidos: List[Dict[str, Any]],
    descartados: List[Dict[str, Any]],
    min_keep_frac: float,
    i: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    total = len(mantidos) + len(descartados)
    alvo = math.ceil(total * float(min_keep_frac))
    if len(mantidos) >= alvo:
        return mantidos, descartados
    # Rebaixa threshold: pega top-k dos descartados por score_relato
    descartados_sorted = sorted(descartados, key=lambda x: float(x.get("score_relato") or 0.0), reverse=True)
    take = max(0, alvo - len(mantidos))
    boost = descartados_sorted[:take]
    rest = descartados_sorted[take:]
    return mantidos + boost, rest


# --------------------------------------------------------------------------
# Predição + calibração + score

def _predizer_e_calibrar(
    modelo: Any,
    tokenizer: Any,
    registros: List[Dict[str, Any]],
    iso_model: Any | None,
    alpha: float,
) -> List[Dict[str, Any]]:
    """Roda a predição GLiNER, aplica ISO por entidade e calcula score_relato."""
    out: List[Dict[str, Any]] = []

    for sample in registros:
        try:
            rec = predict_record(modelo, tokenizer, sample, labels=None)
        except Exception as e:
            # Em caso de falha de predição, devolve registro marcado
            rec = dict(sample)
            rec.setdefault("entities", [])
            rec["erro_predicao"] = str(e)

        # Aplica Isotonic Regression às entidades (escreve em score_ts por compatibilidade)
        try:
            rec = add_score_iso_to_record(rec, iso_model=iso_model, score_in="score", score_out="score_ts")
        except Exception as e:
            rec["erro_iso"] = str(e)

        # Adiciona score_relato por entidade (opcional; útil para debug/traço)
        try:
            rec = add_score_relato_por_entidade(rec)
        except Exception:
            pass

        # Score agregado do relato
        try:
            s_rel = compute_score_relato(rec)
        except Exception:
            s_rel = None
        rec["score_relato"] = None if s_rel is None else float(s_rel)

        out.append(rec)

    return out


def _executa_iteracao(
    itcfg: IterationConfig,
    modelo: Any,
    tokenizer: Any,
    registros: List[Dict[str, Any]],
    iso_model: Any | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # 1) Predizer + calibrar + score_relato
    preditos = _predizer_e_calibrar(modelo, tokenizer, registros, iso_model=iso_model, alpha=itcfg.alpha)

    # 2) Filtrar por tau
    mantidos, descartados = _filtra_por_tau(preditos, tau=itcfg.tau)

    # 3) Piso de 10% (ou valor definido)
    mantidos, descartados = _enforce_min_keep_fraction(
        mantidos, descartados, itcfg.min_keep_frac, itcfg.i
    )

    return mantidos, descartados


# --------------------------------------------------------------------------
# Main

def main(
    input_file: str,
    csv_calibracao: Optional[str] = None,
    iteracoes: int = ITERACOES,
    tau: Optional[float] = None,
    tau_seq: Optional[List[float]] = None,
    alpha: float = ALPHA,
    model_base: str = MODEL_BASE,
    saida_dir: str = SAIDA_DIR,
    min_keep_frac: float = MIN_KEEP_FRAC_DEFAULT,
    continue_all: bool = False,
) -> None:
    # 0) Saudações de configuração
    print(
        f"[Main] Config: input='{input_file}', csv='{csv_calibracao}', iter={iteracoes}, model='{model_base}', min_frac={min_keep_frac:.2f}"
    )

    # 1) Lê a base
    registros = _read_json_or_jsonl(input_file)
    print(f"[Main] Registros carregados: {len(registros)}")

    # 2) Ajuste de Isotonic Regression
    # Diretório para diagnósticos da calibração
    diag_dir = Path(saida_dir) / "diagnosticos_calibracao"
    diag_dir.mkdir(parents=True, exist_ok=True)
    if not csv_calibracao:
        csv_calibracao = CSV_CALIBRACAO
    try:
        iso_model = fit_isotonic_from_csv(
            csv_calibracao,
            n_bins=15,
            save_plots_to=diag_dir,
            plots_prefix="global",
        )
        print(f"[Main] Isotonic Regression ajustado a partir de '{csv_calibracao}'.")
    except Exception as e:
        iso_model = None
        print(f"[Main] Aviso: não foi possível ajustar ISO a partir de '{csv_calibracao}'. Prosseguindo sem calibração. Motivo: {e}")

    # 3) Carrega modelo/tokenizer
    res = load_gliner(model_base)
    device = None
    if isinstance(res, tuple):
        if len(res) == 3:
            model, tokenizer, device = res
        elif len(res) == 2:
            model, tokenizer = res
        else:
            raise ValueError(f"load_gliner retornou {len(res)} valores; esperado 2 ou 3")
    else:
        # fallback extremo: alguns wrappers podem retornar apenas o modelo
        model = res
        tokenizer = None
    
    # 4) Sequência de thresholds
    if tau_seq is None:
        tau_seq = TAU_SEQ_DEFAULT.copy()
    if tau is not None:
        # Se usuário passou --tau, usa valor fixo em todas as iterações
        tau_seq = [float(tau) for _ in range(iteracoes)]
    else:
        # Complementa com fallback, se a lista for menor que o número de iterações
        if len(tau_seq) < iteracoes:
            tau_seq = tau_seq + [TAU_FALLBACK] * (iteracoes - len(tau_seq))

    # 5) Loop principal
    saida = Path(saida_dir)
    saida.mkdir(parents=True, exist_ok=True)

    corrente: List[Dict[str, Any]] = list(registros)
    acumulado_mantidos: List[Dict[str, Any]] = []

    for i in range(1, int(iteracoes) + 1):
        tau_iter = float(tau_seq[i - 1])
        itcfg = IterationConfig(
            i=i,
            tau=tau_iter,
            min_keep_frac=min_keep_frac,
        )

        alvo = corrente if (continue_all or i == 1) else corrente
        # Logging
        print(
            f"[Iter {i}] N={len(alvo)} | tau_iter={tau_iter:.4f} | min_frac={min_keep_frac:.2f}"
        )

        mantidos, descartados = _executa_iteracao(
            itcfg, model, tokenizer, alvo, iso_model=iso_model
        )

        acumulado_mantidos.extend(mantidos)
        corrente = descartados

        # Critérios de parada/logs
        print(
            f"[Iter {i}] mantidos={len(mantidos)} descartados={len(descartados)} (acum={len(acumulado_mantidos)})"
        )

        # Dump por iteração
        itdir = saida / f"iter_{i:02d}"
        itdir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(itdir / "mantidos.jsonl", mantidos)
        _write_jsonl(itdir / "descartados.jsonl", descartados)

    # Consolidados
    _write_jsonl(saida / "mantidos_acumulado.jsonl", acumulado_mantidos)
    _write_jsonl(saida / "descartados_finais.jsonl", corrente)
    print("[Main] Pipeline concluído.")


# ----------------------------------------------------------------------------
# CLI

def _parse_tau_seq(arg: Optional[str]) -> Optional[List[float]]:
    if not arg:
        return None
    try:
        vals = [float(x.strip()) for x in arg.split(",") if x.strip()]
        return vals or None
    except Exception:
        raise argparse.ArgumentTypeError("--tau_seq deve ser uma lista separada por vírgulas, ex.: '0.9,0.8,0.7'")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline iterativo de predição GLiNER com calibração (Isotonic Regression) e filtro por tau.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Caminho do arquivo JSON/JSONL de entrada")
    p.add_argument("--csv", dest="csv", default=CSV_CALIBRACAO, help="CSV de comparação p/ calibrar (colunas: Score, Validacao)")
    p.add_argument("--iter", dest="iter", type=int, default=ITERACOES, help="Número de iterações")
    p.add_argument("--tau", dest="tau", type=float, default=None, help="Threshold fixo por iteração (se informado)")
    p.add_argument("--tau_seq", dest="tau_seq", default=None, help="Sequência de thresholds separada por vírgulas (sobrepõe --tau)")
    p.add_argument("--alpha", dest="alpha", type=float, default=ALPHA, help="Alpha para score do relato (se aplicável)")
    p.add_argument("--model", dest="model", default=MODEL_BASE, help="Nome/caminho do modelo GLiNER")
    p.add_argument("--out", dest="out", default=SAIDA_DIR, help="Diretório de saída")
    p.add_argument("--min_frac", dest="min_frac", type=float, default=MIN_KEEP_FRAC_DEFAULT, help="Piso mínimo de mantidos por iteração (fração)")
    p.add_argument("--continue_all", dest="continue_all", action="store_true", help="Reprocessar todos os registros em cada iteração")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()

    main(
        input_file=args.input,
        csv_calibracao=args.csv,
        iteracoes=args.iter,
        tau=args.tau,
        tau_seq=_parse_tau_seq(args.tau_seq),
        alpha=args.alpha,
        model_base=args.model,
        saida_dir=args.out,
        min_keep_frac=args.min_frac,
        continue_all=args.continue_all,
    )
