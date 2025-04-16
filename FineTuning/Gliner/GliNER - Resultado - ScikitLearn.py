import pandas as pd
from sklearn.metrics import classification_report

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
true_labels_dict = load_ner_data("sanity_test.csv")
pred_labels_dict = load_ner_data("entities_output_TOTAL_SanityTest.csv")

# Encontrar sentenças que existem em ambos os arquivos
common_sentences = set(true_labels_dict.keys()) & set(pred_labels_dict.keys())

# Ajustar os tamanhos das listas de labels para serem iguais
flat_true_labels = []
flat_pred_labels = []

for sentence_id in sorted(common_sentences):
    true = true_labels_dict[sentence_id]
    pred = pred_labels_dict[sentence_id]

    if len(true) < 384 and len(pred) < 384:
        max_len = max(len(true), len(pred))
        true += ["O"] * (max_len - len(true))
        pred += ["O"] * (max_len - len(pred))

        flat_true_labels.extend(true)
        flat_pred_labels.extend(pred)

# Gerar relatório de classificação com Scikit-learn
if flat_true_labels:
    print(classification_report(flat_true_labels, flat_pred_labels))
else:
    print("Nenhuma sentença compatível com o critério de tamanho (< 384) entre os arquivos.")