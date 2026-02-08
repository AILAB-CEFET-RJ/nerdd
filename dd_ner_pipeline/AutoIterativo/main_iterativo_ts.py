# main_iterativo_ts.py — pipeline iterativo com Temperature Scaling (TS)
# ----------------------------------------------------------------------------
# O que este script faz
#  1) Lê uma base JSON/JSONL (um registro por linha ou lista) contendo, no mínimo,
#     um campo de texto (ex.: "relato"/"texto"/"descricao").
#  2) Em cada iteração, roda predição com GLiNER (via predicao.py), aplica
#     Temperature Scaling (calibração global) às entidades, calcula o score do
#     relato como a MÉDIA dos scores calibrados das entidades encontradas por
#     linha e filtra por um threshold (tau).
#  3) Garante um piso de seleção: pelo menos MIN_KEEP_FRAC (ex.: 10%) do conjunto
#     de entrada da iteração é mantido, rebaixando o threshold efetivo (top-k por
#     score_relato) quando necessário.
#  4) Salva, por iteração, dois arquivos JSONL: mantidos.jsonl e descartados.jsonl.
#  5) Treino cumulativo: o fine-tuning começa com os mantidos da 1ª iteração e,
#     nas seguintes, acumula os mantidos anteriores (10 épocas + early stopping
#     com paciência=3 — a métrica de ES deve ser o F1 no treino.py).
# ----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Silencia avisos ruidosos
warnings.filterwarnings("ignore")

# --- Dependências locais ------------------------------------------------------
# predicao.py deve fornecer as funções abaixo (interfaces mínimas usadas aqui):
#   load_gliner(model_name_or_path) -> (model, tokenizer)
#   predict_record(model, tokenizer, sample, labels=None) -> record_com_entities
try:
    from predicao import (
        load_gliner,
        predict_record,
    )
except Exception as e:
    raise RuntimeError(
        "Falha ao importar funções de 'predicao.py'. Verifique se o arquivo está no PYTHONPATH."
    ) from e

# calibracao_ts.py deve fornecer:
#   fit_temperature_from_csv(csv_path) -> float (T)
#   add_score_ts_to_record(record, T, score_in='score', score_out='score_ts') -> record
try:
    from calibracao_ts import (
        fit_temperature_from_csv,
        add_score_ts_to_record,
    )
except Exception as e:
    raise RuntimeError(
        "Falha ao importar funções de 'calibracao_ts.py'. Verifique se o arquivo está no PYTHONPATH."
    ) from e

# treino.py: função de fine-tuning
try:
    from treino import treinar_gliner  # assinatura flexível
except Exception as e:
    treinar_gliner = None  # type: ignore
    print(f"[warn] Não foi possível importar 'treinar_gliner' de treino.py: {e}")

# ----------------------------------------------------------------------------
# Parâmetros padrão
CSV_CALIBRACAO = "/mnt/localssd/comparacao_calibracao.csv"
ITERACOES = 5
TAU_SEQ_DEFAULT = [0.90, 0.80, 0.65, 0.50, 0.35]
TAU_FALLBACK = 0.80  # usado quando TAU_SEQ acabar
MODEL_BASE = "best_overall_gliner_model"
SAIDA_DIR = "saida_iterativa_ts"
MIN_KEEP_FRAC_DEFAULT = 0.10  # mínimo de 10% mantidos por iteração
VAL_FRAC_DEFAULT = 0.10        # 10% para validação
SEED_DEFAULT = 42

# ----------------------------------------------------------------------------
# Auxiliares de texto/entidades/score

TEXT_KEYS = ("text", "texto", "relato", "descricao", "description")


def _get_text(r: Dict[str, Any]) -> str:
    for k in TEXT_KEYS:
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _valid_entities(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    ents = r.get("entities") or r.get("ner") or []
    out: List[Dict[str, Any]] = []
    for e in ents:
        s, t = e.get("start"), e.get("end")
        lab = e.get("label") or e.get("type")
        if isinstance(s, int) and isinstance(t, int) and t > s and lab:
            out.append({"start": int(s), "end": int(t), "label": str(lab), **{k: v for k, v in e.items() if k not in {"start", "end", "label"}}})
    return out


def compute_score_relato_mean(rec: Dict[str, Any], score_key: str = "score_ts") -> Optional[float]:
    """MÉDIA dos scores das entidades do registro (conforme solicitado).
    Usa 'score_ts' se existir; caso contrário cai para 'score'.
    """
    ents = _valid_entities(rec)
    vals: List[float] = []
    for e in ents:
        v = e.get(score_key)
        if v is None:
            v = e.get("score")
        if v is not None:
            try:
                vals.append(float(v))
            except Exception:
                pass
    if not vals:
        return None
    return sum(vals) / len(vals)


def add_score_relato_por_entidade(rec: Dict[str, Any], score_key: str = "score_ts") -> Dict[str, Any]:
    sr = compute_score_relato_mean(rec, score_key=score_key)
    if sr is None:
        return rec
    ents = rec.get("entities") or rec.get("ner") or []
    for e in ents:
        try:
            e["score_relato"] = float(sr)
        except Exception:
            e["score_relato"] = sr
    return rec

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


def _read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    # tenta JSON padrão
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # fallback para JSONL
        return _parse_jsonl(text)
    # JSON ok
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("data", "items", "registros", "records"):
            v = obj.get(k)
            if isinstance(v, list):
                return list(v)
        return [obj]
    raise ValueError("Formato JSON inválido para entrada")


def _write_jsonl(items: Iterable[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

# ----------------------------------------------------------------------------
# Filtragem e seleção

@dataclass
class IterationConfig:
    i: int
    tau: float
    min_keep_frac: float
    alpha: float
    saida_dir: Path
    continue_all: bool


def _filtra_por_tau(registros: List[Dict[str, Any]], tau: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    mantidos: List[Dict[str, Any]] = []
    descartados: List[Dict[str, Any]] = []
    for r in registros:
        sr = r.get("score_relato")
        if sr is None:
            descartados.append(r)
        elif sr >= tau:
            mantidos.append(r)
        else:
            descartados.append(r)
    return mantidos, descartados


def _enforce_min_keep_fraction(mantidos: List[Dict[str, Any]], descartados: List[Dict[str, Any]], min_keep_frac: float, iter_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    total = len(mantidos) + len(descartados)
    if total == 0:
        return mantidos, descartados
    alvo = int(total * min_keep_frac)
    if len(mantidos) >= alvo:
        return mantidos, descartados

    # juntar e ordenar por score_relato desc
    unidos = mantidos + descartados
    for r in unidos:
        sr = r.get("score_relato")
        if sr is None:
            r["score_relato"] = float("-inf")
    unidos.sort(key=lambda x: (x.get("score_relato") if x.get("score_relato") is not None else float("-inf")), reverse=True)

    novos_mantidos = unidos[:alvo]
    novos_descartados = unidos[alvo:]

    tau_efetivo = novos_mantidos[-1].get("score_relato") if novos_mantidos else 0.0
    print(f"[Iter {iter_id}] mínimo forçado de {int(min_keep_frac*100)}%: mantidos={len(novos_mantidos)} (τ efetivo≈{tau_efetivo:.4f})")
    return novos_mantidos, novos_descartados

# ----------------------------------------------------------------------------
# Predição + calibração

def _predizer_e_calibrar(modelo: Any, tokenizer: Any, registros: List[Dict[str, Any]], T: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sample in registros:
        rec = dict(sample)
        try:
            rec = predict_record(modelo, tokenizer, rec, labels=None)
        except Exception as e:
            rec.setdefault("entities", [])
            rec["erro_predicao"] = str(e)
        try:
            rec = add_score_ts_to_record(rec, T, score_in="score", score_out="score_ts")
        except Exception as e:
            rec["erro_ts"] = str(e)
        # score do relato = média dos scores calibrados
        sr = compute_score_relato_mean(rec, score_key="score_ts")
        rec["score_relato"] = sr
        rec = add_score_relato_por_entidade(rec, score_key="score_ts")
        out.append(rec)
    return out

# ----------------------------------------------------------------------------
# Treino cumulativo

def _norm_trainable(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    txt = _get_text(rec)
    ents = _valid_entities(rec)
    if not txt or not ents:
        return None
    rr = dict(rec)
    rr["text"] = txt
    rr["entities"] = ents
    rr["ner"] = ents
    return rr


def _dedup_train(list_recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in list_recs:
        if r is None:
            continue
        key = (r.get("text", ""), tuple((e["start"], e["end"], e["label"]) for e in r.get("entities", [])))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _split_train_val(data: List[Dict[str, Any]], val_frac: float) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    if len(data) >= 10:
        n_val = max(1, int(round(val_frac * len(data))))
    else:
        n_val = 0
    random.shuffle(data)
    train_recs = data[:-n_val] if n_val else data
    val_recs = data[-n_val:] if n_val else None
    return train_recs, val_recs


# ----------------------------------------------------------------------------
# Execução de cada iteração

@dataclass
class IterationConfig:
    i: int
    tau: float
    min_keep_frac: float
    saida_dir: Path
    val_frac: float


def _executa_iteracao(itcfg: IterationConfig, modelo: Any, tokenizer: Any, registros: List[Dict[str, Any]], T: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # 1) Predizer + calibrar + score_relato
    preditos = _predizer_e_calibrar(modelo, tokenizer, registros, T=T)

    # 2) Filtrar por tau
    mantidos, descartados = _filtra_por_tau(preditos, tau=itcfg.tau)

    # 3) Piso mínimo (ex.: 10%)
    mantidos, descartados = _enforce_min_keep_fraction(mantidos, descartados, itcfg.min_keep_frac, itcfg.i)

    # 4) Persistência da iteração
    it_dir = itcfg.saida_dir / f"iter_{itcfg.i:02d}"
    _write_jsonl(mantidos, it_dir / "mantidos.jsonl")
    _write_jsonl(descartados, it_dir / "descartados.jsonl")

    print(f"[Iter {itcfg.i}] mantidos={len(mantidos)} descartados={len(descartados)} (tau={itcfg.tau:.4f})")

    return mantidos, descartados

# ----------------------------------------------------------------------------
# MAIN

def main(
    input_file: str,
    csv_calibracao: Optional[str] = None,
    iteracoes: int = ITERACOES,
    tau: Optional[float] = None,
    tau_seq: Optional[List[float]] = None,
    model_base: str = MODEL_BASE,
    saida_dir: str = SAIDA_DIR,
    min_keep_frac: float = MIN_KEEP_FRAC_DEFAULT,
    val_frac: float = VAL_FRAC_DEFAULT,
    cumulative: bool = True,
    seed: int = SEED_DEFAULT,
) -> None:
    random.seed(seed)

    print(
        f"[Main] Config: input='{input_file}', csv='{csv_calibracao}', iter={iteracoes}, "
        f"model='{model_base}', min_frac={min_keep_frac:.2f}, val_frac={val_frac:.2f}, cumulative={cumulative}"
    )

    # 1) Lê a base
    registros = _read_json_or_jsonl(input_file)
    print(f"[Main] Registros carregados: {len(registros)}")

    # 2) Ajuste do Temperature Scaling
    if not csv_calibracao:
        csv_calibracao = CSV_CALIBRACAO
    try:
        T = float(fit_temperature_from_csv(csv_calibracao))
        print(f"[Main] Temperature Scaling ajustado a partir de '{csv_calibracao}' (T={T:.4f}).")
    except Exception as e:
        T = 1.0
        print(f"[Main] [warn] não foi possível ajustar TS por CSV ('{csv_calibracao}'): {e}; usando T=1.0")

    # 3) Carrega GLiNER
    model, tokenizer = load_gliner(model_base)

    # 4) Sequência de taus
    if tau is not None:
        tau_seq_real = [float(tau)] * iteracoes
    elif tau_seq:
        tau_seq_real = list(tau_seq) + [TAU_FALLBACK] * max(0, iteracoes - len(tau_seq))
        tau_seq_real = tau_seq_real[:iteracoes]
    else:
        tau_seq_real = TAU_SEQ_DEFAULT[:]
        if len(tau_seq_real) < iteracoes:
            tau_seq_real += [TAU_FALLBACK] * (iteracoes - len(tau_seq_real))

    # 5) Loop de iterações
    saida = Path(saida_dir)
    saida.mkdir(parents=True, exist_ok=True)

    corrente = list(registros)
    acumulado_mantidos: List[Dict[str, Any]] = []
    acum_train: List[Dict[str, Any]] = []  # para treino cumulativo

    for i in range(1, iteracoes + 1):
        tau_iter = float(tau_seq_real[i - 1])
        itcfg = IterationConfig(
            i=i,
            tau=tau_iter,
            min_keep_frac=min_keep_frac,
            saida_dir=saida,
            val_frac=val_frac,
        )

        alvo = list(registros) if (not cumulative) else (corrente if corrente else [])
        if len(alvo) == 0:
            print(f"[Iter {i}] não há registros para processar. Encerrando.")
            break

        print(f"[Iter {i}] processando {len(alvo)} registros | tau_iter={tau_iter:.4f} | min_frac={min_keep_frac:.2f}")
        mantidos, descartados = _executa_iteracao(itcfg, model, tokenizer, alvo, T)

        # diagnóstico rápido de texto/entidades
        txt_ok = sum(1 for r in mantidos if _get_text(r))
        ents_ok = sum(1 for r in mantidos if _valid_entities(r))
        ents_total = sum(len(_valid_entities(r)) for r in mantidos)
        print(f"[Iter {i}] diagnostico: mantidos={len(mantidos)} | com_texto={txt_ok} | com_entidades={ents_ok} | total_entidades={ents_total}")

        # Treino cumulativo
        candidatos = [_norm_trainable(r) for r in mantidos]
        candidatos = [c for c in candidatos if c]
        if i == 1:
            acum_train = candidatos
        else:
            acum_train = _dedup_train(acum_train + candidatos)
        train_recs, val_recs = _split_train_val(list(acum_train), val_frac)

        dir_modelo = saida / "modelos" / f"iter_{i:02d}"
        if treinar_gliner is None:
            print("[warn] 'treinar_gliner' indisponível — pulei treino.")
        elif len(train_recs) == 0:
            print(f"[Iter {i}] [warn] treino pulado: 0 amostras treináveis no acumulado.")
        else:
            n_val_log = len(val_recs) if val_recs else 0
            print(f"[Iter {i}] Fine-tuning (ACUM): train={len(train_recs)} | val={n_val_log} -> {dir_modelo}")
            # Chamada com 10 épocas e paciência=3; tentamos passar 'eval_metric'='f1' se suportado
            try:
                treinar_gliner(
                    model, train_recs, val_recs, tokenizer, device=None,
                    out_dir=str(dir_modelo), lr=3e-5, wd=0.01,
                    num_epochs=10, paciencia=3,
                    save_checkpoint=False, save_pretrained=False,
                    eval_metric='f1'
                )
            except TypeError:
                # fallback sem eval_metric
                treinar_gliner(
                    model, train_recs, val_recs, tokenizer, device=None,
                    out_dir=str(dir_modelo), lr=3e-5, wd=0.01,
                    num_epochs=10, paciencia=3,
                    save_checkpoint=False, save_pretrained=False,
                )
            except Exception as e:
                print(f"[Iter {i}] [warn] treino falhou: {e}")

        acumulado_mantidos.extend(mantidos)
        corrente = list(descartados) if cumulative else corrente

        if len(corrente) == 0 and cumulative:
            print(f"[Iter {i}] não há mais descartados para processar. Encerrando.")
            break

    # 6) Saídas agregadas
    _write_jsonl(acumulado_mantidos, saida / "final_mantidos.jsonl")
    if cumulative:
        _write_jsonl(corrente, saida / "final_descartados.jsonl")

    print(f"[Main] Concluído. total_mantidos={len(acumulado_mantidos)} | total_descartados={len(corrente)}")

# ----------------------------------------------------------------------------
# CLI

def _parse_tau_seq(arg: Optional[str]) -> Optional[List[float]]:
    if not arg:
        return None
    try:
        vals = [float(x.strip()) for x in arg.split(',') if x.strip()]
        return vals or None
    except Exception:
        return None


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Pipeline iterativo GLiNER + TS (média por relato).")
    ap.add_argument("--input", required=True, help="Caminho do JSON/JSONL de entrada")
    ap.add_argument("--csv", required=False, default=CSV_CALIBRACAO, help="CSV de calibração para TS (Temperature)")
    ap.add_argument("--iter", type=int, default=ITERACOES, help="Número de iterações")
    ap.add_argument("--tau", type=float, default=None, help="Threshold fixo por iteração")
    ap.add_argument("--tau_seq", type=str, default=None, help="Sequência de taus, ex: '0.9,0.8,0.7'")
    ap.add_argument("--alpha", type=float, default=1.2, help="[reservado]")
    ap.add_argument("--model", type=str, default=MODEL_BASE, help="Modelo/base do GLiNER")
    ap.add_argument("--out", type=str, default=SAIDA_DIR, help="Diretório de saída")
    ap.add_argument("--min_frac", type=float, default=MIN_KEEP_FRAC_DEFAULT, help="Piso mínimo de mantidos por iteração (ex.: 0.10)")
    ap.add_argument("--val_frac", type=float, default=VAL_FRAC_DEFAULT, help="Fração para validação no treino cumulativo (ex.: 0.10)")
    ap.add_argument("--no_cumulative", action="store_true", help="Se passado, reprocessa TODOS a cada iteração (sem acumular descartados)")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Random seed")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(
        input_file=args.input,
        csv_calibracao=args.csv,
        iteracoes=args.iter,
        tau=args.tau,
        tau_seq=_parse_tau_seq(args.tau_seq),
        model_base=args.model,
        saida_dir=args.out,
        min_keep_frac=args.min_frac,
        val_frac=args.val_frac,
        cumulative=(not args.no_cumulative),
        seed=args.seed,
    )
