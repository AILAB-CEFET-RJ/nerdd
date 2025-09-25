#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predição de entidades usando GLiNER, sem BIO e sem K-Fold.

Compatível com main_iterativo_ts.py:
- load_gliner(model_name) -> (model, None, device)
- predict_record(model, tokenizer, record) -> record c/ 'entities'
- add_score_ts_to_record(rec, T, ...)  # Temperature Scaling
- compute_score_relato_mean(rec, score_key="score_ts")
- add_score_relato_por_entidade(rec, score_key="score_ts")

Config por ambiente:
- GLINER_LABELS="Person,Organization,Location" (ou a lista que quiser)
- MAIN_THR=0.05
- FALLBACK_THR="0.03,0.01,0.005"
- PRED_TOP_K=100 (opcional; GLiNER pode ignorar se não suportar)
- GLINER_MAX_LEN=384 (apenas informativo)
"""

from __future__ import annotations
import os
import re
import math
from typing import Any, Dict, List, Optional, Tuple

# =========================
# Configs (env)
# =========================
MAIN_THR: float = float(os.getenv("MAIN_THR", "0.05"))
FALLBACK_THR: List[float] = [
    float(x) for x in os.getenv("FALLBACK_THR", "0.03,0.01,0.005").split(",") if x.strip()
]
PRED_TOP_K: int = int(os.getenv("PRED_TOP_K", "100"))
GLINER_LABELS: List[str] = [
    s.strip()
    for s in os.getenv(
        "GLINER_LABELS",
        "Person,Organization,Location,Bairro,Cidade,Logradouro,PontoDeReferencia,Veículo,Placa,Rodovia,Comunidade",
    ).split(",")
    if s.strip()
]
GLINER_MAX_LEN: int = int(os.getenv("GLINER_MAX_LEN", "384"))

# =========================
# Helpers de texto
# =========================
TEXT_KEYS = ("text", "relato", "texto", "descricao", "description")

def _norm_text(rec: Dict[str, Any]) -> str:
    for k in TEXT_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

# =========================
# Fallback heurístico
# =========================
_PLACA_RE = re.compile(r"\b([A-Z]{3}\s?-?\d{4})\b", re.I)

def _fallback_entities_from_text(text: str, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    ents: List[Dict[str, Any]] = []
    if not isinstance(text, str) or not text:
        return ents

    # Placas
    for m in _PLACA_RE.finditer(text):
        ents.append({"start": m.start(), "end": m.end(), "label": "Placa", "score": 0.20})

    # Locais conhecidos no próprio registro (só se estiverem dentro do texto)
    for k, lab in (
        ("bairroLocal", "Bairro"),
        ("cidadeLocal", "Cidade"),
        ("logradouroLocal", "Logradouro"),
        ("pontodeReferenciaLocal", "PontoDeReferencia"),
    ):
        v = rec.get(k)
        if isinstance(v, str):
            vv = v.strip().strip(",")
            if vv:
                lo = text.lower().find(vv.lower())
                if lo != -1:
                    ents.append({"start": lo, "end": lo + len(vv), "label": lab, "score": 0.15})

    return ents

# =========================
# Calibração / Scoring
# =========================
def _clip01(p: float, eps: float = 1e-6) -> float:
    return max(eps, min(1.0 - eps, float(p)))

def _logit(p: float) -> float:
    p = _clip01(p)
    return math.log(p / (1.0 - p))

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def add_score_ts_to_record(
    rec: Dict[str, Any],
    T: float,
    score_in: str = "score",
    score_out: str = "score_ts",
) -> Dict[str, Any]:
    """Aplica Temperature Scaling em cada entidade do registro."""
    ents = rec.get("entities") or rec.get("ner") or []
    out = []
    for e in ents:
        s = float(e.get(score_in, 0.20))
        try:
            z = _logit(s) / float(T)
            s_ts = _sigmoid(z)
        except Exception:
            s_ts = float(s)
        ee = dict(e)
        ee[score_out] = float(s_ts)
        out.append(ee)
    if rec.get("entities") is not None:
        rec["entities"] = out
    else:
        rec["ner"] = out
    return rec

def compute_score_relato_mean(rec: Dict[str, Any], score_key: str = "score_ts") -> float:
    """Média dos scores (calibrados se disponíveis) das entidades do relato."""
    ents = rec.get("entities") or rec.get("ner") or []
    vals: List[float] = []
    for e in ents:
        if score_key in e and e[score_key] is not None:
            vals.append(float(e[score_key]))
        elif "score" in e and e["score"] is not None:
            vals.append(float(e["score"]))
    return float(sum(vals) / len(vals)) if vals else 0.0

def add_score_relato_por_entidade(rec: Dict[str, Any], score_key: str = "score_ts") -> Dict[str, Any]:
    """Replica o score médio do relato para cada entidade (campo 'score_relato')."""
    sr = compute_score_relato_mean(rec, score_key=score_key)
    ents = rec.get("entities") or rec.get("ner") or []
    new_ents = []
    for e in ents:
        ee = dict(e)
        ee["score_relato"] = float(sr)
        new_ents.append(ee)
    if rec.get("entities") is not None:
        rec["entities"] = new_ents
    else:
        rec["ner"] = new_ents
    rec["score_relato"] = float(sr)
    return rec

# =========================
# Carregamento GLiNER
# =========================
def load_gliner(model_name: str):
    """
    Carrega GLiNER e retorna (model, None, device) para compatibilidade.
    Não usa BIO, não usa K-Fold.
    """
    import torch
    from gliner import GLiNER

    # Evita caminhos com accelerate
    os.environ.setdefault("TRANSFORMERS_NO_ACCELERATE", "1")

    model = GLiNER.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        print(f"[load_gliner] gliner={model_name} | max_len={GLINER_MAX_LEN}")
    except Exception:
        print(f"[load_gliner] gliner={model_name}")
    # Retorna None no lugar do tokenizer para manter assinatura
    return model, None, device

# =========================
# Predição por registro (sem BIO)
# =========================
def predict_record(model: Any, tokenizer: Any, record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Usa GLiNER.predict_entities(text, labels, threshold) para obter spans (char offsets).
    Sem BIO, sem K-Fold. Aplica fallback se vazio.
    """
    rec = dict(record)
    text = _norm_text(rec)
    # Trunca de forma conservadora (GLiNER já lida com chunk, mas evitamos exageros)
    if isinstance(text, str) and len(text) > 20000:
        text = text[:20000]
    rec["text"] = text

    ents: List[Dict[str, Any]] = []

    if hasattr(model, "predict_entities"):
        # tenta thresholds em cascata
        for thr in [MAIN_THR] + FALLBACK_THR:
            try:
                # Algumas versões aceitam top_k; se não aceitar, cai no except e re-chama sem top_k
                try:
                    preds = model.predict_entities(text, labels=GLINER_LABELS, threshold=thr, top_k=PRED_TOP_K)
                except TypeError:
                    preds = model.predict_entities(text, labels=GLINER_LABELS, threshold=thr)

                ents_tmp: List[Dict[str, Any]] = []
                for p in preds or []:
                    s0 = int(p.get("start", -1))
                    e0 = int(p.get("end", -1))
                    lab = p.get("label") or p.get("type")
                    sc = float(p.get("score", 0.0))
                    if s0 >= 0 and e0 > s0 and isinstance(lab, str) and lab:
                        ents_tmp.append({"start": s0, "end": e0, "label": lab, "score": sc})
                if ents_tmp:
                    ents = ents_tmp
                    break
            except Exception as e:
                rec["erro_predicao"] = str(e)

    # Fallback heurístico se nada vier
    if not ents:
        ents = _fallback_entities_from_text(text, rec)

    # Garante 'score' numérico
    for e in ents:
        if "score" not in e or e["score"] is None:
            e["score"] = 0.20

    rec["entities"] = ents
    return rec