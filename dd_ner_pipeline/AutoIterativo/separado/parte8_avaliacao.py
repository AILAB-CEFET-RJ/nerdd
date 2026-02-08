#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Avaliação rápida do GLiNER em arquivo JSONL no formato:
{"text": "...", "spans": [{"start":..., "end":..., "label":"..."}]}

Saídas:
- <out_dir>/predicoes.jsonl
- <out_dir>/classification_report.txt
- <out_dir>/metrics.json
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# ------------------------ IO ------------------------

def read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        # JSONL
        items: List[Dict[str, Any]] = []
        for i, ln in enumerate(txt.splitlines(), 1):
            ln = ln.strip()
            if not ln:
                continue
            try:
                items.append(json.loads(ln))
            except json.JSONDecodeError as e:
                raise ValueError(f"Falha no JSONL (linha {i}): {e}")
        return items
    if isinstance(obj, list):
        return obj
    return [obj]

def write_jsonl(items: Iterable[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def write_text(text: str, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# ------------------------ Métricas (span exato) ------------------------

def to_span_set(spans: List[Dict[str, Any]]) -> set[tuple[int,int,str]]:
    s = set()
    for e in spans or []:
        try:
            s.add((int(e["start"]), int(e["end"]), str(e["label"])))
        except Exception:
            pass
    return s

def prf1_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}

def classification_report_per_label(golds: List[List[Dict[str,Any]]],
                                   preds: List[List[Dict[str,Any]]]) -> Tuple[str, Dict[str, Any]]:
    # Conjuntos por amostra
    gold_sets = [to_span_set(g) for g in golds]
    pred_sets = [to_span_set(p) for p in preds]

    # Coleta rótulos distintos (dos golds)
    labels = sorted({lab for gs in gold_sets for (_,_,lab) in gs}) or []

    # Contagens por rótulo
    per_label = {}
    micro_tp = micro_fp = micro_fn = 0

    for lab in labels:
        tp = fp = fn = 0
        support = 0
        for gset, pset in zip(gold_sets, pred_sets):
            g_lab = {(s,e,l) for (s,e,l) in gset if l == lab}
            p_lab = {(s,e,l) for (s,e,l) in pset if l == lab}
            tp += len(g_lab & p_lab)
            fp += len(p_lab - g_lab)
            fn += len(g_lab - p_lab)
            support += len(g_lab)
        micro_tp += tp; micro_fp += fp; micro_fn += fn
        scores = prf1_from_counts(tp, fp, fn)
        per_label[lab] = {**scores, "support": support}

    # Micro
    micro = prf1_from_counts(micro_tp, micro_fp, micro_fn)

    # Macro-F1: média dos F1 por rótulo (se não houver labels, 0.0)
    macro_f1 = sum(per_label[lab]["f1"] for lab in per_label)/len(per_label) if per_label else 0.0

    # Gera um txt estilo classification_report
    lines = []
    lines.append("label                precision   recall   f1-score   support")
    for lab in labels:
        r = per_label[lab]
        lines.append(f"{lab:<20}  {r['precision']:>9.3f}  {r['recall']:>7.3f}  {r['f1']:>9.3f}  {r['support']:>8d}")
    lines.append("")
    lines.append(f"micro avg           {micro['precision']:>9.3f}  {micro['recall']:>7.3f}  {micro['f1']:>9.3f}  {sum(per_label[l]['support'] for l in labels):>8d}")
    lines.append(f"macro f1: {macro_f1:.3f}")
    report_txt = "\n".join(lines)

    metrics = {
        "per_label": per_label,
        "micro": micro,
        "macro_f1": macro_f1,
        "labels": labels,
        "overall_support": sum(per_label[l]["support"] for l in labels) if labels else 0
    }
    return report_txt, metrics

# ------------------------ Predição GLiNER ------------------------

def predict_all(model, tokenizer, records: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """Retorna cópias dos registros com campo 'entities' previsto."""
    out: List[Dict[str,Any]] = []
    import predicao as pred  # já importado antes; reuso
    for rec in records:
        base = {"text": rec.get("text","")}
        try:
            pr = pred.predict_record(model, tokenizer, base)
        except Exception as e:
            pr = dict(base); pr["entities"] = []; pr["erro_predicao"] = str(e)
        out.append(pr)
    return out

# ------------------------ Main ------------------------

def main():
    p = argparse.ArgumentParser(description="Avaliação GLiNER em JSONL (span-level).")
    p.add_argument("--input", required=True, help="Caminho do gliner_teste_sanidade.json")
    p.add_argument("--model", required=True, help="Diretório/nome do modelo GLiNER (ex.: best_overall_gliner_model)")
    p.add_argument("--out_dir", default="avaliacao_sanidade", help="Diretório de saída")
    p.add_argument("--labels", default="Person,Location,Organization", help="Rótulos GLiNER (separados por vírgula)")
    p.add_argument("--thr", type=float, default=0.05, help="Threshold principal de predição")
    p.add_argument("--fallback_thr", default="0.03,0.01,0.005", help="Cascata de thresholds fallback (vírgula)")
    p.add_argument("--top_k", type=int, default=100, help="top_k (se suportado pela versão do GLiNER)")
    args = p.parse_args()

    # Ajusta parâmetros do predicao.py DEPOIS de importar (safe: são variáveis de módulo)
    import predicao as pred
    pred.GLINER_LABELS = [s.strip() for s in args.labels.split(",") if s.strip()]
    pred.MAIN_THR = float(args.thr)
    pred.FALLBACK_THR = [float(x) for x in args.fallback_thr.split(",") if x.strip()]
    pred.PRED_TOP_K = int(args.top_k)

    # Carrega modelo
    model, tokenizer, device = pred.load_gliner(args.model)

    # Lê dados
    dados = read_json_or_jsonl(args.input)

    # Normaliza gold (usa campo "spans")
    golds: List[List[Dict[str,Any]]] = []
    for r in dados:
        spans = r.get("spans") or []
        golds.append([
            {"start": int(s["start"]), "end": int(s["end"]), "label": str(s["label"])}
            for s in spans if {"start","end","label"} <= set(s.keys())
        ])

    # Predição
    previstos = predict_all(model, tokenizer, dados)
    preds_only = [pr.get("entities") or [] for pr in previstos]

    # Salva predições
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(previstos, out_dir / "predicoes.jsonl")

    # Métricas + relatório
    report_txt, metrics = classification_report_per_label(golds, preds_only)
    write_text(report_txt, out_dir / "classification_report.txt")
    write_json(metrics, out_dir / "metrics.json")

    # Exibe F1 no stdout
    print(report_txt)
    print(f"\nF1 (micro): {metrics['micro']['f1']:.4f}")
    print(f"F1 (macro): {metrics['macro_f1']:.4f}")
    print(f"Salvo em: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
