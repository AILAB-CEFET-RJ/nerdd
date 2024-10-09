import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report
import io


teste = "Bases/test_data.csv"
predicao = "entities_output.csv"




# Carregar os CSVs
csv_real_sent = pd.read_csv(teste, sep=';', names=['Sentence', 'Classificacao'])
csv_pred = pd.read_csv(predicao, sep=',', names=['Palavra', 'Classificacao'])


sentences = csv_real_sent['Sentence']

# Para armazenar tokens e classificações
tokens = []
classificacoes = []

# Processar cada sentença
for sentence in sentences:
    # Dividir a sentença em tokens e classificações
    palavras_classificacoes = sentence.strip().split('\n')
    
    for pc in palavras_classificacoes:
        if pc.strip():  # Verifica se a linha não está vazia
            palavra, classificacao = pc.split(';')
            tokens.append(palavra.strip())
            classificacoes.append(classificacao.strip())

# Criar um novo DataFrame com os tokens e classificações
csv_real = pd.DataFrame({
    'Palavra': tokens,
    'Classificacao': classificacoes
})

# Filtrar linhas que contêm '@@'
csv_real = csv_real[csv_real['Palavra'] != "\"\n\""]
csv_pred = csv_pred[csv_pred['Palavra'] != "\"\n\""]

csv_real = csv_real[csv_real['Classificacao'] != 'O']
csv_pred = csv_pred[csv_pred['Classificacao'] != 'Label']

# Resetar os índices para garantir alinhamento
csv_real.reset_index(drop=True, inplace=True)
csv_pred.reset_index(drop=True, inplace=True)

# Agora, garantir que ambas as listas tenham o mesmo comprimento
csv_pred = csv_pred[csv_pred.index.isin(csv_real.index)]

# Comparar as classificações
y_true = csv_real['Classificacao'].values
y_pred = csv_pred['Classificacao'].values

# Garantir que y_true e y_pred tenham o mesmo tamanho
y_true = y_true[:len(y_pred)]

# prompt: Colocando em minusculo


# Colocando em minúsculo as classificações
y_true = [x.lower() for x in y_true]
y_pred = [x.lower() for x in y_pred]


# Calcular a acurácia
#accuracy = accuracy_score(y_true, y_pred)

# Calcular a precisão para cada classe
#precision = precision_score(y_true, y_pred, average=None, labels=['person', 'location','organization'])
#recall = recall_score(y_true, y_pred, average=None, labels=['person', 'location','organization'])

# Gerar a matriz de confusão
#conf_matrix = confusion_matrix(y_true, y_pred, labels=['person', 'location','organization'])

# Exibir os resultados
#print(f'Acurácia: {accuracy}')
#print(f'Precisão: {precision}')
#print(f'Revocação: {recall}')
#print('Matriz de confusão:')
#print(conf_matrix)

# Criar um DataFrame para a matriz de confusão com rótulos (labels)
#labels=['person', 'location','organization']
#conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

#print(conf_matrix_df)

# Gerar o relatório de classificação
report = classification_report(y_true, y_pred)

# Exibir o relatório
print(report)