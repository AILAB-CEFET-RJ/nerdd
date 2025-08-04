import json
import pandas as pd
import unicodedata

# Função para normalizar texto
def normalizar(texto):
    if not texto:
        return ""
    texto = unicodedata.normalize('NFKD', texto)
    texto = texto.encode('ASCII', 'ignore').decode('utf-8')
    return texto.lower().strip()

# Leitura do JSONL
with open('Corpus_grande_pseudolabel_score0.json', 'r', encoding='utf-8') as f:
    data = [json.loads(linha) for linha in f]

# Filtrar relatos com entidades
data_com_entidades = [item for item in data if 'entities' in item and item['entities']]

# Lista para registrar alterações de score
alteracoes_score = []

# Ajustar scores com base em localização
for item in data_com_entidades:
    referencias = {
        normalizar(item.get('bairroLocal')),
        normalizar(item.get('logradouroLocal')),
        normalizar(item.get('cidadeLocal')),
        normalizar(item.get('pontodeReferenciaLocal'))
    }

    for entidade in item['entities']:
        texto_normalizado = normalizar(entidade['text'])
        score_antigo = entidade['score']
        if texto_normalizado in referencias:
            score_novo = min(score_antigo * 1.2, 1.0)
            if score_novo != score_antigo:
                alteracoes_score.append({
                    "entidade": entidade['text'],
                    "score_antigo": score_antigo,
                    "score_novo": score_novo
                })
            entidade['score'] = score_novo

# Criar DataFrame geral (completa)
df = pd.DataFrame(data_com_entidades)


with open('Corpus_Grande_Pseudolabel_Threshold.json', 'w', encoding='utf-8') as f_out:
    for linha in data_com_entidades:
        json.dump(linha, f_out, ensure_ascii=False)
        f_out.write('\n')
# Salvar corpus completo
df.to_csv('Corpus_Grande_Pseudolabel_Threshold.csv', index=False)
with open('Corpus_Grande_Pseudolabel_Threshold06.json', 'w', encoding='utf-8') as f_out:
    for linha in data_com_entidades:
        # Manter somente entidades com score > 0.6
        entidades_filtradas = [ent for ent in linha['entities'] if ent['score'] > 0.6]
        if entidades_filtradas:
            linha_filtrada = linha.copy()
            linha_filtrada['entities'] = entidades_filtradas
            json.dump(linha_filtrada, f_out, ensure_ascii=False)
            f_out.write('\n')

# Criar DataFrame com alterações de score
df_alteracoes = pd.DataFrame(alteracoes_score)
df_alteracoes.to_csv('Alteracoes_Metadados.csv', index=False)

print("✅ Arquivos salvos:")
print("- corpus_comparado.csv")
print("- corpus_comparado.json")
print("- alteracoes_score.csv")
