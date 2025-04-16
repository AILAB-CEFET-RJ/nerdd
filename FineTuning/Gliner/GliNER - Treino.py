import json
import random
import torch
import matplotlib.pyplot as plt
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollator
import os
import gc
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

torch.cuda.empty_cache()
gc.collect()


def compute_metrics(p):
    preds = p.predictions.argmax(-1) if hasattr(p.predictions, 'argmax') else p.predictions
    labels = p.label_ids

    f1 = f1_score(labels, preds, average="weighted")
    return {"f1": f1}


# Caminho para o conjunto de dados
data_path = "subset_100.json"

# Carregar os dados
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f'Dataset size: {len(data)}')

# Embaralhar os dados aleatoriamente
random.shuffle(data)
print('Dataset is shuffled...')

def split_long_sentences(dataset, max_length=384):
    split_data = []
    
    for sample in dataset:
        words = sample["tokenized_text"]
        ner_annotations = sample["ner"]

        if len(words) > max_length:
            for i in range(0, len(words), max_length):
                # Filtra entidades que estão dentro do intervalo atual
                new_ner = [
                    [start - i, end - i, label]
                    for start, end, label in ner_annotations
                    if start >= i and end < i + max_length
                ]
                
                split_data.append({
                    "tokenized_text": words[i:i + max_length],
                    "ner": new_ner
                })
        else:
            split_data.append(sample)

    return split_data

# Divisão do dataset
split_index = int(len(data) * 0.9)
train_dataset = data[:split_index]
test_dataset = data[split_index:]

# Aplicar divisão para evitar truncamento
train_dataset = split_long_sentences(train_dataset)
test_dataset = split_long_sentences(test_dataset)

print(f"Total training samples after splitting: {len(train_dataset)}")
print(f"Total test samples after splitting: {len(test_dataset)}")

# Configuração do ambiente
torch.backends.cudnn.benchmark = True  # Melhor desempenho para redes convolucionais
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar modelo
model = GLiNER.from_pretrained("urchade/gliner_small").to(device)
model.data_processor.max_len = 1024  # Aumente para o limite do modelo
model.data_processor.transformer_tokenizer.model_max_length = 1024  # Ajuste para o limite desejado
model.data_processor.transformer_tokenizer.padding_side = "right"  # Pode ajudar no truncamento correto

print(f'Model max token length: {model.data_processor.transformer_tokenizer.model_max_length}')

# Data Collator
data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

# Hiperparâmetros
epochs = 200
batch_size = 4

training_args = TrainingArguments(
    fp16=True,
    output_dir="models-adjust",
    learning_rate=5e-6,
    weight_decay=0.01,
    others_lr=1e-5,
    others_weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    evaluation_strategy="steps",
    save_steps=40000,
    save_total_limit=10,
    dataloader_num_workers=2,
    use_cpu=not torch.cuda.is_available(),
    report_to="none",
    compute_metrics=compute_metrics
)

# Treinador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=model.data_processor.transformer_tokenizer,
    data_collator=data_collator,
)

# Treinar o modelo
train_result = trainer.train()
print("Training complete!")

""""

# ----------- Avaliação com F1-score no conjunto de validação -----------
print("Running evaluation on validation set...")

# Desativar gradiente para inferência
model.eval()
all_preds = []
all_labels = []

for sample in test_dataset:
    text = " ".join(sample["tokenized_text"])
    true_entities = set(tuple(ent) for ent in sample["ner"])  # Convert to tuple for set operations
    pred_entities = set(tuple(ent) for ent in model.predict_entities(text))

    all_preds.append(pred_entities)
    all_labels.append(true_entities)

# Converter em listas binárias para avaliação micro/macro
from sklearn.metrics import precision_recall_fscore_support

# Obter todas as entidades possíveis
all_entity_types = set()
for ents in all_labels + all_preds:
    for _, _, label in ents:
        all_entity_types.add(label)
all_entity_types = sorted(list(all_entity_types))

def to_multilabel_vector(entities_set, label_list):
    vec = [0] * len(label_list)
    for _, _, label in entities_set:
        if label in label_list:
            vec[label_list.index(label)] = 1
    return vec

y_true_bin = [to_multilabel_vector(labs, all_entity_types) for labs in all_labels]
y_pred_bin = [to_multilabel_vector(preds, all_entity_types) for preds in all_preds]

# Calcular F1-score macro
precision, recall, f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="macro")

print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1-score (macro): {f1:.4f}")

# Salvar logs de treinamento
with open("training_logs.json", "w", encoding="utf-8") as f:
    json.dump(trainer.state.log_history, f, indent=4)
print("Logs de treinamento salvos em 'training_logs.json'")


# Caminho para o arquivo de logs
log_path = "training_logs.json"

# Carregar os logs
with open(log_path, "r", encoding="utf-8") as f:
    logs = json.load(f)

# Filtrar apenas logs que têm tanto "step" quanto "loss"
filtered_logs = [log for log in logs if "step" in log and "loss" in log]

# Extrair steps e losses
steps = [log["step"] for log in filtered_logs]
losses = [log["loss"] for log in filtered_logs]

# Plotar gráfico
plt.figure(figsize=(8, 5))
plt.plot(steps, losses, marker="o", linestyle="-", label="Training Loss", color="b")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss Over Steps")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Salvar imagem
plt.savefig("training_loss.png", dpi=300)
print("✅ Gráfico de perda salvo como 'training_loss.png'")

# ============================
# Geração do Gráfico de Perda
# ============================
# Capturar logs
logs = trainer.state.log_history

# Coletar f1-score ao longo do tempo (usando sua própria avaliação manual no final)
# Ou, se for adicionar callback depois, você pode logar direto com eval_f1
f1_scores = []
eval_steps = []

for i, log in enumerate(logs):
    if "eval_loss" in log and "step" in log:
        step = log["step"]
        # Você pode usar outro campo aqui se estiver logando F1: log.get("eval_f1", ...)
        # Por enquanto, adicionamos o valor manual que foi calculado no final
        # Exemplo fictício: você quer adicionar um ponto por epoch
        eval_steps.append(step)
        f1_scores.append(None)  # Aqui você pode armazenar os f1 acumulados se tiver

# Como você já calculou f1 no final (macro), podemos plotar apenas esse valor por agora
# Exemplo básico:
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), [f1]*epochs, marker='o', label="F1-score (macro)")
plt.title("F1-score ao longo das épocas")
plt.xlabel("Épocas")
plt.ylabel("F1-score")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("f1_scores.png")
plt.close()

print("Gráfico de F1-score salvo em 'f1_scores.png'.")

# Filtrar apenas logs que têm tanto "step" quanto "loss"
filtered_logs = [log for log in logs if "step" in log and "loss" in log]

# Extrair os valores
steps = [log["step"] for log in filtered_logs]
train_losses = [log["loss"] for log in filtered_logs]


plt.figure(figsize=(8, 5))
plt.plot(steps, train_losses, marker="o", linestyle="-", label="Training Loss", color="b")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss Over Steps")
plt.legend()
plt.grid()
plt.savefig("training_loss.png", dpi=300)
print("Gráfico de perda salvo como 'training_loss.png'")

"""