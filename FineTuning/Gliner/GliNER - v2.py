import csv
from collections import defaultdict
from gliner import GLiNER

# Carregando modelo treinado
trained_model = GLiNER.from_pretrained("models-adjust/checkpoint-160400", load_tokenizer=True)
trained_model.data_processor.transformer_tokenizer.model_max_length = 1024
print(f'Model max token length: {trained_model.data_processor.transformer_tokenizer.model_max_length}')

# Nome do arquivo CSV de entrada
# csv_input_file = 'Bases/test_data_filtered.csv'
csv_input_file = 'diferencas.csv'



# Lendo o arquivo CSV e agrupando tokens por sentença
sentences = defaultdict(list)
entities_by_sentence = defaultdict(list)

with open(csv_input_file, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        sentence_id = int(row['sentence_id'])
        word = row['word']
        label = row['label']
        sentences[sentence_id].append(word)
        entities_by_sentence[sentence_id].append({"word": word, "label": label})

# Filtrando sentenças que tenham no máximo 384 tokens
filtered_sentences = {k: v for k, v in sentences.items() if len(v) <= 384}

# Reconstruindo as sentenças
texts2 = {k: " ".join(words).replace(" .", ".") for k, words in filtered_sentences.items()}
print(f"Total de sentenças após filtragem: {len(texts2)}")

# Nome do arquivo CSV de saída
#csv_output_file = 'Bases/entities_output_TOTAL_MD2.csv'
csv_output_file = 'entities_output_TOTAL_SanityTest_v2.csv'

labels = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]

# Dicionário para armazenar previsões por sentença
predicted_labels = {k: ["O"] * len(v) for k, v in filtered_sentences.items()}

# Realizando a predição das entidades
for sentence_id, sentence in texts2.items():
    entities = trained_model.predict_entities(sentence, labels, threshold=0)
    
    # Criando um mapeamento para palavras da sentença
    words = filtered_sentences[sentence_id]

    for entity in entities:
        entity_text = entity["text"]
        entity_label = entity["label"]
        
        # Encontrando a posição da palavra na sentença
        for i, word in enumerate(words):
            if word == entity_text:
                predicted_labels[sentence_id][i] = entity_label

# Escrevendo o CSV mantendo a estrutura original
with open(csv_output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['sentence_id', 'word', 'label'])

    for sentence_id, words in filtered_sentences.items():
        for word, label in zip(words, predicted_labels[sentence_id]):
            writer.writerow([sentence_id, word, label])

print(f'Dados salvos com sucesso no arquivo {csv_output_file}')