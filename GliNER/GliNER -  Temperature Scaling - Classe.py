import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

INPUT_FILE = "Corpus_grande_pseudolabel_score0.json"
OUTPUT_FILE = "Corpus_grande_pseudolabel_score0_TS_Classe.json"

def load_data(path):
    scores = []
    labels = []
    truths = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            for ent in sample.get("entities", []):
                scores.append(ent["score"])
                labels.append(ent["label"])
                truths.append(1 if ent["score"] >= 0.9 else 0)  # Marca como true ou false
    return np.array(scores), labels, np.array(truths)

def temperature_scaling(scores, truths):
    logits = np.log(scores / (1 - scores + 1e-15))  # Evita log(0)
    logits = logits.reshape(-1, 1)
    truths = truths.reshape(-1)

    model = LogisticRegression(solver="lbfgs")
    model.fit(logits, truths)
    T = 1 / model.coef_[0][0]  # Inverso do peso = temperatura
    return T

def apply_temperature_scaling(scores, T):
    logits = np.log(scores / (1 - scores + 1e-15))
    scaled_logits = logits / T
    calibrated_scores = 1 / (1 + np.exp(-scaled_logits))
    return calibrated_scores

def plot_calibration(y_true, y_prob, nome="Temperature Scaling"):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label=nome)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Score Previsão")
    plt.ylabel("Probabilidade Real")
    plt.title("Curva de Calibração")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("curva_calibracao.png", dpi=300)
    plt.close()

def salvar_calibrado(input_path, output_path, calibrated_scores):
    idx = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            sample = json.loads(line)
            for ent in sample.get("entities", []):
                ent["score_original"] = ent["score"]
                ent["score_calibrado"] = float(calibrated_scores[idx])
                ent["true"] = 1 if ent["score_original"] >= 0.9 else 0
                idx += 1
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

# === Execução ===
scores, labels, truths = load_data(INPUT_FILE)
T = temperature_scaling(scores, truths)
print(f"Temperatura encontrada: {T:.4f}")

calibrated_scores = apply_temperature_scaling(scores, T)
print(f"Brier antes: {brier_score_loss(truths, scores):.4f}")
print(f"Brier depois: {brier_score_loss(truths, calibrated_scores):.4f}")

plot_calibration(truths, scores, nome="Antes da Calibração")
plot_calibration(truths, calibrated_scores, nome="Depois da Calibração")
salvar_calibrado(INPUT_FILE, OUTPUT_FILE, calibrated_scores)
print(f"Arquivo calibrado salvo em: {OUTPUT_FILE}")
