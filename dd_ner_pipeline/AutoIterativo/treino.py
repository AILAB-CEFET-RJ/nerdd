# /home/gustavo/treino.py
from __future__ import annotations
import os, math, json, random
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F

# ===================== util: reproducibilidade =====================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ===================== helpers de texto / spans ====================
_TEXT_KEYS = ("text","relato","texto","descricao","description")
def _get_text(rec: Dict[str,Any]) -> str:
    for k in _TEXT_KEYS:
        v = rec.get(k)
        if isinstance(v,str) and v.strip():
            return v
    return ""

def _whitespace_tokenize_with_char_spans(text: str) -> Tuple[List[str], List[Tuple[int,int]]]:
    tokens, spans = []
    start = 0
    n = len(text)
    out_tokens: List[str] = []
    out_spans: List[Tuple[int,int]] = []
    while start < n:
        # pula espaços
        while start < n and text[start].isspace():
            start += 1
        if start >= n:
            break
        end = start
        while end < n and not text[end].isspace():
            end += 1
        out_tokens.append(text[start:end])
        out_spans.append((start,end))
        start = end
    return out_tokens, out_spans

def _char_to_token_spans(txt: str, ents: List[Dict[str,Any]]) -> List[List[Any]]:
    """Converte spans de caracteres -> spans de tokens [i0, i1, label]."""
    tokens, tspans = _whitespace_tokenize_with_char_spans(txt)
    ner: List[List[Any]] = []
    for e in ents or []:
        s = int(e.get("start", -1))
        ed = int(e.get("end", -1))
        lab = e.get("label")
        if not (isinstance(lab,str) and lab):
            continue
        if s < 0 or ed <= s or s >= len(txt):
            continue
        ed = min(ed, len(txt))
        st_i = None
        en_i = None
        for i,(ts,te) in enumerate(tspans):
            # qualquer sobreposição conta
            if ts < ed and te > s:
                if st_i is None:
                    st_i = i
                en_i = i
        if st_i is not None and en_i is not None and st_i <= en_i:
            ner.append([int(st_i), int(en_i), lab])
    return tokens, ner

# ===================== métrica: F1 por spans (exato) =====================
def _f1_from_span_lists(pred_spans_list: List[List[Dict[str,Any]]],
                        gold_spans_list: List[List[Dict[str,Any]]]) -> float:
    # transforma listas em conjuntos de tuplas (start,end,label)
    tp = fp = fn = 0
    for pred, gold in zip(pred_spans_list, gold_spans_list):
        pset = set((int(x["start"]),int(x["end"]),str(x["label"])) for x in pred)
        gset = set((int(x["start"]),int(x["end"]),str(x["label"])) for x in gold)
        tp += len(pset & gset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    if (2*tp + fp + fn) == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2*precision*recall/(precision+recall)

# ===================== dataset GLiNER (spanlabel) =====================
def _make_spanlabel_dataset(recs: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    ds: List[Dict[str,Any]] = []
    for r in recs:
        txt = _get_text(r)
        if not txt:
            continue
        ents = r.get("entities") or r.get("ner") or []
        toks, ner = _char_to_token_spans(txt, ents)
        if ner:
            ds.append({"tokenized_text": toks, "ner": ner})
    return ds

# ===================== treino GLiNER =====================
@dataclass
class _TrainConf:
    batch_size: int = 8
    lr: float = 3e-5
    wd: float = 0.01
    max_epochs: int = 10
    patience: int = 3
    threshold_eval: float = 0.5
    num_workers: int = 0  # deixa 0 para evitar problemas

def _train_gliner_spanlabel(model: Any,
                            train_recs: List[Dict[str,Any]],
                            val_recs: List[Dict[str,Any]],
                            out_dir: str,
                            conf: _TrainConf):
    from gliner.data_processing.collator import DataCollator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # monta dataset em spanlabel
    ds_train = _make_spanlabel_dataset(train_recs)
    ds_val   = _make_spanlabel_dataset(val_recs) if val_recs else []

    if not ds_train:
        print("[treino] nada a treinar em GLiNER (0 exemplos com ner).")
        return

    # conjunto de labels (apenas para logs e eval)
    labels = sorted({lab for ex in ds_train for _,_,lab in ex["ner"]})
    print(f"[treino] SpanLabels: {labels}")

    collate = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

    train_dl = DataLoader(ds_train, batch_size=conf.batch_size, shuffle=True,
                          num_workers=conf.num_workers, collate_fn=collate, pin_memory=False)
    val_dl   = DataLoader(ds_val, batch_size=conf.batch_size, shuffle=False,
                          num_workers=conf.num_workers, collate_fn=collate, pin_memory=False) if ds_val else None

    optim = AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.wd)

    best_f1 = -1.0
    best_state = None
    patience = 0

    # prepara gold de validação para F1 (char spans)
    def _to_char_spans(sample: Dict[str,Any]) -> Tuple[str, List[Dict[str,Any]]]:
        # reconstrói o texto só para métricas (tokens unidos por espaço)
        text = " ".join(sample["tokenized_text"])
        # cria spans de caracteres consistentes com esse texto reconstruído:
        offs = []
        cur = 0
        for tok in sample["tokenized_text"]:
            start = text.find(tok, cur)
            end = start + len(tok)
            offs.append((start,end))
            cur = end + 1
        gold = []
        for st,en,lab in sample["ner"]:
            if st < len(offs) and en < len(offs):
                gold.append({"start":offs[st][0],"end":offs[en][1],"label":lab})
        return text, gold

    val_texts = []
    val_gold  = []
    if ds_val:
        for ex in ds_val:
            t,g = _to_char_spans(ex)
            val_texts.append(t)
            val_gold.append(g)

    for epoch in range(1, conf.max_epochs+1):
        model.train()
        run_loss = 0.0
        nb = 0
        for batch in train_dl:
            nb += 1
            batch = {k: v.to(device) if hasattr(v,"to") else v for k,v in batch.items()}
            out = model(**batch)  # GLiNER devolve loss já correta
            loss = out.loss
            run_loss += float(loss.detach().cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad()
        tr_loss = run_loss/max(1,nb)

        if not val_dl:
            print(f"[treino] epoch={epoch} train_loss={tr_loss:.4f}")
            continue

        # avaliação por F1 com predict_entities (spanlabel)
        model.eval()
        with torch.no_grad():
            preds_all: List[List[Dict[str,Any]]] = []
            for text in val_texts:
                try:
                    preds = model.predict_entities(text, labels=labels, threshold=conf.threshold_eval)
                except TypeError:
                    preds = model.predict_entities(text, labels=labels, threshold=conf.threshold_eval)
                # normaliza formato
                out_spans = []
                for p in preds or []:
                    s = int(p.get("start",-1)); e = int(p.get("end",-1)); lab = p.get("label") or p.get("type")
                    if s>=0 and e>s and isinstance(lab,str) and lab:
                        out_spans.append({"start":s,"end":e,"label":lab})
                preds_all.append(out_spans)
            f1 = _f1_from_span_lists(preds_all, val_gold)

        print(f"[treino] epoch={epoch} train_loss={tr_loss:.4f} | val_f1={f1:.4f} (patience {patience}/{conf.patience})")

        improved = f1 > best_f1
        if improved:
            best_f1 = f1
            patience = 0
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= conf.patience:
                print(f"[treino] Early stopping na epoch {epoch}. Melhor F1={best_f1:.4f}")
                break

    # restaura melhor estado (se houver) e salva
    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(out_dir, exist_ok=True)
    try:
        model.save_pretrained(out_dir)
    except Exception as e:
        print(f"[treino] warn: não consegui save_pretrained: {e}")

# ===================== API pública =====================
def treinar_gliner(model: Any,
                   train_recs: List[Dict[str,Any]],
                   val_recs: List[Dict[str,Any]] | None = None,
                   tokenizer: Any = None,
                   device: Any = None,
                   out_dir: str = "modelo_saidax",
                   num_epochs: int = 10,
                   paciencia: int = 3,
                   save_checkpoint: bool = False,
                   save_pretrained: bool = True,
                   batch_size: int = 8,
                   lr: float = 3e-5,
                   wd: float = 0.01,
                   weight_decay: float | None = None,
                   eval_metric: str | None = None,
                   dl_num_workers: int | None = None,
                   **kwargs):
    """
    Se tokenizer for None (ou modelo for GLiNER), treina em modo 'spanlabel' com GLiNER.
    Aceita lr, wd/weight_decay, batch_size, num_epochs, paciencia, dl_num_workers, etc.
    """
    set_seed(42)

    # normaliza weight decay
    if weight_decay is not None:
        wd = float(weight_decay)

    # número de workers (evita problemas de prefetch_factor quando 0)
    num_workers = int(dl_num_workers) if dl_num_workers is not None else 0

    # --- MODO GLiNER (spanlabel) ---
    if (tokenizer is None) or hasattr(model, "predict_entities"):
        conf = _TrainConf(
            batch_size=int(batch_size),
            lr=float(lr),
            wd=float(wd),
            max_epochs=int(num_epochs),
            patience=int(paciencia),
            num_workers=num_workers,
        )
        _train_gliner_spanlabel(model, train_recs, val_recs or [], out_dir, conf)
        return

    # --- (fallback) modo antigo BIO, desativado neste projeto ---
    raise RuntimeError("Treino BIO desativado neste projeto: use GLiNER (spanlabel).")
