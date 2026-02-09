from sklearn.metrics import f1_score


def f1_score_from_span_lists(pred_spans_list, gold_spans_list, average="macro"):
    """Compute F1 from exact-match span sets (start, end, label)."""
    true_labels = []
    pred_labels = []

    for pred_spans, gold_spans in zip(pred_spans_list, gold_spans_list):
        gold_set = set((span["start"], span["end"], span["label"]) for span in gold_spans)
        pred_set = set((span["start"], span["end"], span["label"]) for span in pred_spans)

        all_spans = gold_set.union(pred_set)
        for span in all_spans:
            true_labels.append(1 if span in gold_set else 0)
            pred_labels.append(1 if span in pred_set else 0)

    if not true_labels and not pred_labels:
        return 1.0
    if not true_labels or not pred_labels:
        return 0.0

    return f1_score(true_labels, pred_labels, average=average, zero_division=0)


def compute_f1_by_threshold(model, dataset, threshold, entity_labels):
    """Predict entities and compute macro F1 at a confidence threshold."""
    all_preds = []
    gold_spans = [spans for _, spans in dataset]

    for text, _ in dataset:
        preds = model.predict_entities(text, labels=entity_labels, threshold=threshold)
        filtered = [pred for pred in preds if pred["label"] in entity_labels]
        all_preds.append(filtered)

    return f1_score_from_span_lists(all_preds, gold_spans, average="macro")
