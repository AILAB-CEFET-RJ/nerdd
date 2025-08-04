import json
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from scipy.optimize import minimize

# Caminhos fixos
ENTRADA = 'Corpus_grande_pseudolabel_score0.json'        # Substitua pelo seu nome de arquivo original
SAIDA = 'Corpus_grande_pseudolabel_score0_TS.json'       # Arquivo de saída com scores calibrados

# Modelinho de temperature scaling
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

def temperature_scaling(scores, labels):
    scores = torch.tensor(scores, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    # converter probabilidade para logit
    logits = torch.log(scores / (1 - scores))
    model = TemperatureScaler()

    def loss_fn(temp):
        temp = torch.tensor(temp, requires_grad=False)
        scaled_logits = logits / temp
        loss = nn.BCEWithLogitsLoss()(scaled_logits, labels)
        return loss.item()

    # minimizar a loss para encontrar temperatura ideal
    res = minimize(loss_fn, x0=[1.0], bounds=[(0.5, 5.0)])
    temperatura_otima = res.x[0]

    # aplicar temperatura e obter novos scores
    scaled_logits = logits / temperatura_otima
    calibrated_scores = torch.sigmoid(scaled_logits).tolist()

    return calibrated_scores, temperatura_otima

# Leitura e coleta de scores
dados = []
scores_originais = []
labels_simulados = []

with open(ENTRADA, 'r', encoding='utf-8') as f:
    for linha in f:
        exemplo = json.loads(linha)
        for ent in exemplo.get('entities', []):
            score = ent['score']
            label_simulado = 1 if score >= 0.5 else 0  # substitua por ground truth real se tiver
            scores_originais.append(score)
            labels_simulados.append(label_simulado)
        dados.append(exemplo)

# Calibrar os scores
scores_calibrados, temperatura = temperature_scaling(scores_originais, labels_simulados)
print(f"\nTemperatura ótima encontrada: {temperatura:.4f}\n")

# Substituir scores
indice = 0
for exemplo in dados:
    for ent in exemplo.get('entities', []):
        ent['score_calibrado'] = round(scores_calibrados[indice], 6)
        indice += 1

# Salvar arquivo calibrado
with open(SAIDA, 'w', encoding='utf-8') as f:
    for exemplo in dados:
        f.write(json.dumps(exemplo, ensure_ascii=False) + '\n')

print(f"Arquivo salvo com scores calibrados: {SAIDA}")
