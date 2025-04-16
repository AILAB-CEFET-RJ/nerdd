import json
import random
import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset
import os
import csv

"""
train_path = "train_data.json"

with open(train_path, "r",encoding="utf-8") as f:
    data = json.load(f)

print('Dataset size:', len(data))

random.shuffle(data)
print('Dataset is shuffled...')

train_dataset = data[:int(len(data)*0.9)]
test_dataset = data[int(len(data)*0.9):]

print('Dataset is splitted...')

os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model = GLiNER.from_pretrained("urchade/gliner_small")

# use it for better performance, it mimics original implementation but it's less memory efficient
data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

# Optional: compile model for faster training
model.to(device)
print("done")

# calculate number of epochs
num_steps = 500
batch_size = 8
data_size = len(train_dataset)
num_batches = data_size // batch_size
#num_epochs = max(1, num_steps // num_batches)
num_epochs = 200

training_args = TrainingArguments(
    output_dir="models-adjust",
    learning_rate=5e-6,
    weight_decay=0.01,
    others_lr=1e-5,
    others_weight_decay=0.01,
    lr_scheduler_type="linear", #cosine
    warmup_ratio=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    evaluation_strategy="steps",
    save_steps = 10000,
    save_total_limit=10,
    dataloader_num_workers = 0,
    use_cpu = False,
    report_to="none",
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=model.data_processor.transformer_tokenizer,
    data_collator=data_collator,
)

trainer.train() 

"""

import json
import csv
from gliner import GLiNER

# Carregando modelo treinado
trained_model = GLiNER.from_pretrained("models-adjust/checkpoint-78800", load_tokenizer=True)

trained_model.data_processor.transformer_tokenizer.model_max_length = 1024  # Ajuste para o limite desejado

print(f'Model max token length: {trained_model.data_processor.transformer_tokenizer.model_max_length}')


# Lendo o arquivo JSON
with open('Bases/test_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Função para reconstruir sentenças (sem destacar entidades)
def reconstruct_sentence(data_item):
    sentence = " ".join(data_item["tokenized_text"])
    sentence = sentence.replace(" .", ".")  # Ajusta a pontuação
    return sentence

# Aplicando a função a cada item de data e armazenando as sentenças no array 'texts'
texts2 = [reconstruct_sentence(item) for item in data]

print(f"Total de sentenças: {len(texts2)}")

# Nome do arquivo CSV
csv_file = 'Bases/entities_output_TOTAL_MD.csv'
labels = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]

# Lista para armazenar entidades reais do JSON
real_entities = []

# Extraindo entidades reais do JSON
for item in data:
    if "entities" in item:
        for entity in item["entities"]:
            real_entities.append({
                "text": entity["text"],
                "label": entity["label"]
            })

# Conjunto para armazenar entidades previstas
predicted_entities = set()

# Abrindo o arquivo CSV para escrita
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Escrevendo o cabeçalho
    writer.writerow(['sentence_id', 'word', 'label'])
    
    # Iterando sobre os textos para fazer predições
    for sentence_id, den in enumerate(texts2):
        # Fazendo a predição de entidades
        entities = trained_model.predict_entities(den, labels, threshold=0)
        
        # Armazenando as entidades previstas e salvando no CSV
        for entity in entities:
            predicted_entities.add((sentence_id, entity["text"], entity["label"]))
            writer.writerow([sentence_id, entity["text"], entity["label"]])

# Identificando entidades reais que não foram previstas
missed_entities = [
    entity for entity in real_entities
    if (entity["text"], entity["label"]) not in [(text, label) for _, text, label in predicted_entities]
]

# Exibindo entidades não previstas
print(f"Entidades não previstas ({len(missed_entities)}):")
for missed in missed_entities:
    print(f"Texto: {missed['text']}, Rótulo: {missed['label']}")

print(f'Dados salvos com sucesso no arquivo {csv_file}')
