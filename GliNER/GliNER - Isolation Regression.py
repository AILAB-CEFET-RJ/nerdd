import json
import numpy as np
from sklearn.isotonic import IsotonicRegression

# Arquivos
ENTRADA = 'Corpus_grande_pseudolabel_score0.json'
SAIDA = 'Corpus_grande_pseudolabel_score0_IR.json'

# Coletar todos os scores
scores = []
with open(ENTRADA, 'r', encoding='utf-8') as f:
    for linha in f:
        exemplo = json.loads(linha)
        for ent in exemplo.get('entities', []):
            scores.append(ent['score'])

scores = np.array(scores)

# Definir limites de percentil para simular labels
lim_inf = np.percentile(scores, 30)
lim_sup = np.percentile(scores, 70)

# Simular labels: 1 para top 30%, 0 para bottom 30%
scores_treino = []
labels_simulados = []

with open(ENTRADA, 'r', encoding='utf-8') as f:
    for linha in f:
        exemplo = json.loads(linha)
        for ent in exemplo.get('entities', []):
            score = ent['score']
            if score <= lim_inf:
                scores_treino.append(score)
                labels_simulados.append(0)
            elif score >= lim_sup:
                scores_treino.append(score)
                labels_simulados.append(1)
            # scores intermediários são ignorados

# Verificar equilíbrio
print(f"→ Labels simulados: {labels_simulados.count(0)} negativos | {labels_simulados.count(1)} positivos")

if len(set(labels_simulados)) < 2:
    raise ValueError("⚠️ Falha na simulação de labels: ajuste os percentis para garantir exemplos positivos e negativos.")

# Treinar regressão isotônica
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(scores_treino, labels_simulados)

# Aplicar a calibração em todos os scores
with open(ENTRADA, 'r', encoding='utf-8') as fin, open(SAIDA, 'w', encoding='utf-8') as fout:
    for linha in fin:
        exemplo = json.loads(linha)
        for ent in exemplo.get('entities', []):
            ent['score_calibrado'] = round(float(ir.predict([ent['score']])[0]), 6)
        fout.write(json.dumps(exemplo, ensure_ascii=False) + '\n')

print(f"\n✅ Arquivo salvo com scores calibrados via Isotonic Regression: {SAIDA}")
