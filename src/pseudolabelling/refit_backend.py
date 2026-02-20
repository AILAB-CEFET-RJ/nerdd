from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


def set_seed(seed: int = 42):
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_gliner(model_name: str):
    import torch
    from gliner import GLiNER

    os.environ.setdefault("TRANSFORMERS_NO_ACCELERATE", "1")
    model = GLiNER.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, None, device


_TEXT_KEYS = ("text", "relato", "texto", "descricao", "description")


def _get_text(record: Dict[str, Any]) -> str:
    for key in _TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _whitespace_tokenize_with_char_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    if not isinstance(text, str):
        text = str(text or "")
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    length = len(text)
    while cursor < length:
        while cursor < length and text[cursor].isspace():
            cursor += 1
        if cursor >= length:
            break
        start = cursor
        while cursor < length and not text[cursor].isspace():
            cursor += 1
        end = cursor
        tokens.append(text[start:end])
        spans.append((start, end))
    return tokens, spans


def _char_to_token_spans(text: str, entities: List[Dict[str, Any]]) -> Tuple[List[str], List[List[Any]]]:
    tokens, token_spans = _whitespace_tokenize_with_char_spans(text)
    ner = []
    for entity in entities or []:
        try:
            start_char = int(entity.get("start", -1))
            end_char = int(entity.get("end", -1))
        except Exception:
            continue
        label = entity.get("label")
        if not isinstance(label, str) or not label:
            continue
        if start_char < 0 or end_char <= start_char or start_char >= len(text):
            continue
        end_char = min(end_char, len(text))
        start_token = None
        end_token = None
        for idx, (tok_start, tok_end) in enumerate(token_spans):
            if tok_start < end_char and tok_end > start_char:
                if start_token is None:
                    start_token = idx
                end_token = idx
        if start_token is not None and end_token is not None:
            ner.append([int(start_token), int(end_token), label])
    return tokens, ner


def _f1_from_span_lists(predicted_spans: List[List[Dict[str, Any]]], gold_spans: List[List[Dict[str, Any]]]) -> float:
    tp = fp = fn = 0
    for predicted, gold in zip(predicted_spans, gold_spans):
        pred_set = set((int(x["start"]), int(x["end"]), str(x["label"])) for x in predicted)
        gold_set = set((int(x["start"]), int(x["end"]), str(x["label"])) for x in gold)
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _make_spanlabel_dataset(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dataset = []
    for record in records:
        text = _get_text(record)
        if not text:
            continue
        entities = record.get("entities") or record.get("ner") or []
        tokenized_text, ner = _char_to_token_spans(text, entities)
        if ner:
            dataset.append({"tokenized_text": tokenized_text, "ner": ner})
    return dataset


@dataclass
class _TrainConf:
    batch_size: int = 8
    lr: float = 3e-5
    wd: float = 0.01
    max_epochs: int = 10
    patience: int = 3
    threshold_eval: float = 0.5
    num_workers: int = 0


def _train_gliner_spanlabel(model: Any, train_recs: List[Dict[str, Any]], val_recs: List[Dict[str, Any]], out_dir: str, conf: _TrainConf):
    from gliner.data_processing.collator import DataCollator
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds_train = _make_spanlabel_dataset(train_recs)
    ds_val = _make_spanlabel_dataset(val_recs) if val_recs else []
    if not ds_train:
        print("[refit] no trainable examples after span conversion")
        return

    labels = sorted({label for example in ds_train for _, _, label in example["ner"]})
    collate = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)
    train_loader = DataLoader(
        ds_train,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=conf.num_workers,
        collate_fn=collate,
        pin_memory=False,
    )
    val_loader = (
        DataLoader(
            ds_val,
            batch_size=conf.batch_size,
            shuffle=False,
            num_workers=conf.num_workers,
            collate_fn=collate,
            pin_memory=False,
        )
        if ds_val
        else None
    )
    optimizer = AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.wd)

    best_f1 = -1.0
    best_state = None
    patience_counter = 0

    val_texts = []
    val_gold = []
    if ds_val:
        for example in ds_val:
            text = " ".join(example["tokenized_text"])
            offsets = []
            cursor = 0
            for token in example["tokenized_text"]:
                start = text.find(token, cursor)
                end = start + len(token)
                offsets.append((start, end))
                cursor = end + 1
            gold = []
            for start_idx, end_idx, label in example["ner"]:
                if start_idx < len(offsets) and end_idx < len(offsets):
                    gold.append({"start": offsets[start_idx][0], "end": offsets[end_idx][1], "label": label})
            val_texts.append(text)
            val_gold.append(gold)

    for epoch in range(1, conf.max_epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in train_loader:
            batches += 1
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss
            running_loss += float(loss.detach().cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        train_loss = running_loss / max(1, batches)
        if not val_loader:
            print(f"[refit] epoch={epoch} train_loss={train_loss:.4f}")
            continue

        model.eval()
        with torch.no_grad():
            predicted = []
            for text in val_texts:
                preds = model.predict_entities(text, labels=labels, threshold=conf.threshold_eval)
                normalized = []
                for span in preds or []:
                    start = int(span.get("start", -1))
                    end = int(span.get("end", -1))
                    label = span.get("label") or span.get("type")
                    if start >= 0 and end > start and isinstance(label, str) and label:
                        normalized.append({"start": start, "end": end, "label": label})
                predicted.append(normalized)
            f1 = _f1_from_span_lists(predicted, val_gold)

        print(
            f"[refit] epoch={epoch} train_loss={train_loss:.4f} | "
            f"val_f1={f1:.4f} (patience {patience_counter}/{conf.patience})"
        )

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= conf.patience:
                print(f"[refit] early stopping at epoch {epoch}. best_f1={best_f1:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)


def treinar_gliner(
    model: Any,
    train_recs: List[Dict[str, Any]],
    val_recs: List[Dict[str, Any]] | None = None,
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
    **kwargs,
):
    set_seed(42)
    if weight_decay is not None:
        wd = float(weight_decay)
    num_workers = int(dl_num_workers) if dl_num_workers is not None else 0

    conf = _TrainConf(
        batch_size=int(batch_size),
        lr=float(lr),
        wd=float(wd),
        max_epochs=int(num_epochs),
        patience=int(paciencia),
        num_workers=num_workers,
    )
    _train_gliner_spanlabel(model, train_recs, val_recs or [], out_dir, conf)
