import pandas as pd
from sklearn.metrics import classification_report

# Lendo os arquivos CSV
entities_df = pd.read_csv("Bases/test_data.csv")
sentences_df = pd.read_csv("entities_output.csv")

# Ajustando os rótulos para letras maiúsculas para consistência
entities_df['Label'] = entities_df['Label'].str.upper()
sentences_df['Label'] = sentences_df['Label'].str.upper()

# Garantindo que estamos comparando somente palavras comuns nos dois arquivos
entities_df = entities_df.rename(columns={"Label": "true_label"})
sentences_df = sentences_df.rename(columns={"Label": "predicted_label"})

# Fazer a junção apenas com base na ordem original dos arquivos
min_length = min(len(entities_df), len(sentences_df))
entities_df = entities_df.head(min_length)
sentences_df = sentences_df.head(min_length)

# Comparação palavra a palavra
y_true = entities_df['true_label'].tolist()
y_pred = sentences_df['predicted_label'].tolist()

# Gerar o relatório de classificação
report = classification_report(y_true, y_pred, labels=['OBJECT', 'PERSON', 'LOCATION', 'ORGANIZATION'], zero_division=0)
print(report)
