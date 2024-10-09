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


train_path = "resultado-treino.json"

with open(train_path, "r") as f:
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
num_epochs = max(1, num_steps // num_batches)

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
    save_steps = 100,
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

trained_model = GLiNER.from_pretrained("models-adjust/checkpoint-593", load_tokenizer=True)

# Lendo o arquivo JSON
with open('resultado-teste.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Função para reconstruir sentenças (sem destacar entidades)
def reconstruct_sentence(data_item):
    sentence = " ".join(data_item["tokenized_text"])
    sentence = sentence.replace(" .", ".")  # Ajusta a pontuação
    return sentence

# Aplicando a função a cada item de data e armazenando as sentenças no array 'texts'
texts2 = [reconstruct_sentence(item) for item in data]

# Supondo que 'texts', 'trained_model', e 'labels' já estão definidos

# Nome do arquivo CSV
csv_file = 'entities_output.csv'
labels = ["person", "organization","location"]

# Abra o arquivo em modo de escrita
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Escreva o cabeçalho do CSV
    writer.writerow(['Entity', 'Label'])
    
    # Itera sobre os textos e faz a predição
    for den in texts2:
        # Faz a predição de entidades
        entities = trained_model.predict_entities(den, labels, threshold=0.5)
        
        # Escreve cada entidade e seu rótulo no arquivo CSV
        for entity in entities:
            writer.writerow([entity["text"], entity["label"]])

print(f'Dados salvos com sucesso no arquivo {csv_file}')