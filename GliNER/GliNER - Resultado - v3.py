import json
import csv
import re
from gliner import GLiNER
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# Função para dividir textos longos em fragmentos
def split_text(text, max_length=384, tokenizer=None):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [tokenizer.decode(tokens[i:i + max_length]) for i in range(0, len(tokens), max_length)]

# Função para filtrar entidades inválidas
def clean_entities(entities, original_text):
    cleaned = []
    for ent in entities:
        if ent["text"].strip() in ["[", "]", "CLS", "SEP"]:
            continue
        if ent["start"] < 0 or ent["end"] > len(original_text):
            continue
        if not re.match(r"^[\wÀ-ÿ\s\-\.\']+$", ent["text"]):
            continue
        cleaned.append(ent)
    return cleaned

# Carregar modelo
trained_model = GLiNER.from_pretrained("models-adjust\\workspace-20072025\\best_model", load_tokenizer=True)
tokenizer = trained_model.data_processor.transformer_tokenizer
tokenizer.model_max_length = 1024

# Labels usadas
labels = ["Person", "Location", "Organization"]

# Caminhos
input_path = "gliner_teste_sanidade.json"
output_path_jsonl = "gliner_teste_sanidade_v2_resultado_metadados_pseudolabel.json"
output_path_csv = "gliner_teste_sanidade_v2_resultado_metadados_pseudolabel.csv"

# Ler os textos do arquivo JSON
texts = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        texts.append(item["text"])


def calibrate_thresholds(gt_data, pred_data, labels, thresholds_to_try=None):
    if thresholds_to_try is None:
        thresholds_to_try = [round(x * 0.05, 2) for x in range(0, 21)]  # de 0.0 a 1.0

    best_thresholds = {}

    for label in labels:
        best_f1 = 0
        best_threshold = 0.5  # valor padrão
        for thresh in thresholds_to_try:
            y_true, y_pred, per_class_counts = evaluate(gt_data, pred_data, score_threshold=thresh)
            tp = per_class_counts.get((label, "TP"), 0)
            fp = per_class_counts.get((label, "FP"), 0)
            fn = per_class_counts.get((label, "FN"), 0)

            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        best_thresholds[label] = {
            "best_threshold": best_threshold,
            "best_f1": best_f1
        }
        print(f"🔧 Threshold ideal para '{label}': {best_threshold} com F1: {best_f1:.4f}")

    return best_thresholds

# Predição em lotes
def predict_in_batches(model, texts, labels, batch_size=8, threshold=None):
    all_predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        preds = model.batch_predict_entities(batch, labels, threshold=threshold)
        all_predictions.extend(preds)
    return all_predictions

# Fazer predições para cada fragmento
all_predictions = []
index = 0
for text in texts:
    split_parts = split_text(text, max_length=384, tokenizer=tokenizer)
    part_predictions = predict_in_batches(trained_model, split_parts, labels, batch_size=4, threshold=0.0)

    merged_entities = []
    for part_text, entities in zip(split_parts, part_predictions):
        cleaned = clean_entities(entities, part_text)
        offset = text.find(part_text)  # Ajuste seguro
        for ent in cleaned:
            ent_fixed = ent.copy()
            ent_fixed["start"] += offset
            ent_fixed["end"] += offset
            merged_entities.append(ent_fixed)
        index += 1

    all_predictions.append(merged_entities)

# Salvar em JSONL
with open(output_path_jsonl, "w", encoding="utf-8") as f_out:
    for text, entities in zip(texts, all_predictions):
        json.dump({"text": text, "entities": entities}, f_out, ensure_ascii=False)
        f_out.write("\n")

print("✅ Predições limpas e salvas com sucesso em JSONL e CSV.")

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def extract_entities(entry, key, score_threshold=None):
    if key not in entry:
        return set()

    ents = set()
    for ent in entry[key]:
        if score_threshold is not None and "score" in ent:
            if isinstance(score_threshold, dict):
                # Threshold por entidade
                ent_thresh = score_threshold.get(ent["label"], 0.5)
                if ent["score"] < ent_thresh:
                    continue
            else:
                # Threshold global
                if ent["score"] < score_threshold:
                    continue
        ents.add((ent["start"], ent["end"], ent["label"]))
    return ents


def evaluate(gt_data, pred_data, score_threshold=None):
    if len(gt_data) != len(pred_data):
        print(f"Erro: arquivos com número diferente de entradas ({len(gt_data)} vs {len(pred_data)})")
        sys.exit(1)

    y_true = []
    y_pred = []
    per_class_counts = Counter()

    for gt_entry, pred_entry in zip(gt_data, pred_data):
        gt_ents = extract_entities(gt_entry, "spans")
        pred_ents = extract_entities(pred_entry, "entities", score_threshold=score_threshold)

        # Verdadeiros Positivos
        for ent in pred_ents & gt_ents:
            y_true.append(ent[2])
            y_pred.append(ent[2])
            per_class_counts[(ent[2], "TP")] += 1

        # Falsos Positivos
        for ent in pred_ents - gt_ents:
            y_true.append("None")
            y_pred.append(ent[2])
            per_class_counts[(ent[2], "FP")] += 1

        # Falsos Negativos
        for ent in gt_ents - pred_ents:
            y_true.append(ent[2])
            y_pred.append("None")
            per_class_counts[(ent[2], "FN")] += 1

    return y_true, y_pred, per_class_counts

def print_per_class(per_class_counts):
    labels = set(label for (label, _) in per_class_counts)
    print("\nResumo por classe:")
    for label in sorted(labels):
        tp = per_class_counts.get((label, "TP"), 0)
        fp = per_class_counts.get((label, "FP"), 0)
        fn = per_class_counts.get((label, "FN"), 0)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        print(f"{label:10} | TP: {tp:2} FP: {fp:2} FN: {fn:2} | Precision: {precision:.2f} Recall: {recall:.2f} F1: {f1:.2f}")


def save_classification_report(y_true, y_pred, per_class_counts, filename="classification_report.txt"):
    labels = sorted(set(y_true + y_pred) - {"None"})

    # Gerar relatório do scikit-learn
    report = classification_report(y_true, y_pred, labels=labels, digits=2)

    # Calcular acurácia
    acc = accuracy_score(y_true, y_pred)

    # Calcular F1-macro
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro")

    # Gerar resumo por classe
    class_summary = ["\nResumo por classe:"]
    for label in labels:
        tp = per_class_counts.get((label, "TP"), 0)
        fp = per_class_counts.get((label, "FP"), 0)
        fn = per_class_counts.get((label, "FN"), 0)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        class_summary.append(f"{label:12} | TP: {tp:<4} FP: {fp:<4} FN: {fn:<4} | Precision: {precision:.2f} Recall: {recall:.2f} F1: {f1:.2f}")

    # Salvar tudo no arquivo
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Relatório geral (classification_report):\n")
        f.write(report + "\n")
        f.write(f"\nAcurácia total: {acc:.4f}\n")
        f.write(f"F1-macro: {f1_macro:.4f}\n")
        f.write("\n".join(class_summary))
    
    print(f"\n📊 Acurácia total: {acc:.4f}")
    print(f"🎯 F1-macro: {f1_macro:.4f}")
    print(f"📄 Relatório salvo em: {filename}")

if __name__ == "__main__":
    gt_file = "gliner_teste_sanidade.json"
    pred_file = "gliner_teste_sanidade_v2_resultado_metadados_pseudolabel.json"

    gt_data = load_jsonl(gt_file)
    pred_data = load_jsonl(pred_file)

    # Labels do seu problema
    labels = ["Person", "Location", "Organization"]

    # 🔍 Calibrar thresholds por entidade
    best_thresholds = calibrate_thresholds(gt_data, pred_data, labels)

    # Salvar thresholds calibrados
    with open("calibrated_metadados_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(best_thresholds, f, indent=2, ensure_ascii=False)

    # ✅ Avaliação final com thresholds calibrados por entidade
    thresholds_dict = {k: v["best_threshold"] for k, v in best_thresholds.items()}
    y_true, y_pred, per_class_counts = evaluate(gt_data, pred_data, score_threshold=thresholds_dict)

    print("\n🎯 Avaliação final com thresholds calibrados por entidade:")
    print(classification_report(y_true, y_pred, labels=sorted(list(set(y_true) - {"None"}))))

    print_per_class(per_class_counts)

    save_classification_report(
        y_true,
        y_pred,
        per_class_counts,
        filename="classification_report_com_thresholds_metadados_calibrados.txt"
    )
