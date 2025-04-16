import pandas as pd
from seqeval.metrics import classification_report

def load_ner_data(csv_path):
    df = pd.read_csv(csv_path)

    # Ordenar por sentence_id para evitar desalinhamento
    df = df.sort_values(by=["sentence_id"]).reset_index(drop=True)

    # Remover IDs específicos
    df = df[~df["sentence_id"].isin([17, 61])]

    # Agrupar os rótulos por sentence_id
    grouped = df.groupby("sentence_id")["label"].apply(list)

    return grouped.to_dict()  # Retorna um dicionário {sentence_id: [labels]}

# Carregar os arquivos CSV
# true_labels_dict = load_ner_data("Bases/test_data_filtered.csv")  # Caminho correto
# pred_labels_dict = load_ner_data("Bases/entities_output_TOTAL_MD2.csv")   # Caminho correto
true_labels_dict = load_ner_data("diferencas.csv")  # Caminho correto
pred_labels_dict = load_ner_data("entities_output_TOTAL_SanityTest_v2.csv")   # Caminho correto



# Encontrar sentenças que existem em ambos os arquivos
common_sentences = set(true_labels_dict.keys()) & set(pred_labels_dict.keys())

# Ajustar os tamanhos das listas de labels para serem iguais
filtered_true_labels = []
filtered_pred_labels = []

for sentence_id in sorted(common_sentences):  # Ordena para manter consistência
    true = true_labels_dict[sentence_id]
    pred = pred_labels_dict[sentence_id]

    # Verificar se o tamanho da sentença é menor que 384
    if len(true) < 384 and len(pred) < 384:
        max_len = max(len(true), len(pred))  # Determina o tamanho máximo da sentença

        # Preenche com "O" para igualar os tamanhos
        true += ["O"] * (max_len - len(true))
        pred += ["O"] * (max_len - len(pred))

        filtered_true_labels.append(true)  # Mantemos a estrutura de listas de listas
        filtered_pred_labels.append(pred)

# Gerar relatório de classificação com seqeval
if filtered_true_labels:
    print(classification_report(filtered_true_labels, filtered_pred_labels))
else:
    print("Nenhuma sentença compatível com o critério de tamanho (< 384) entre os arquivos.")
