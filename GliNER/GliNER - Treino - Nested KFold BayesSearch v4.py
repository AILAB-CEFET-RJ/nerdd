import json
import random
import torch
from gliner import GLiNER
from gliner.data_processing.collator import DataCollator
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score
import numpy as np
from collections import defaultdict
import copy
import gc
from tqdm import tqdm
# from skopt import BayesSearchCV # Removido para simplificar a integração direta no loop
# from skopt.space import Real # Removido

# ==================== Configurações de Reproducibilidade =====================
def set_seed(seed):
    """Define as sementes para reproducibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 42
set_seed(SEED)

# ==================== Utilitários =====================

def tokenize_with_spans(text):
    """
    Tokeniza o texto por espaços e retorna os tokens e seus spans de caracteres.
    Esta função é crucial para o alinhamento correto das anotações.
    """
    tokens = []
    token_spans = []
    start = 0
    while start < len(text):
        # Ignora espaços em branco no início
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            break
        end = start
        # Encontra o fim do token (até o próximo espaço)
        while end < len(text) and not text[end].isspace():
            end += 1
        token = text[start:end]
        tokens.append(token)
        token_spans.append((start, end))
        start = end
    return tokens, token_spans

def process_sample(sample):
    """
    Processa uma amostra de dados, tokenizando o texto e mapeando os spans de caracteres
    das entidades para spans de tokens.
    """
    text = sample["text"]
    spans = sample.get("spans", [])
    tokens, token_spans = tokenize_with_spans(text)
    ner = []
    for span in spans:
        start_char = span["start"]
        end_char = span["end"]
        label = span["label"]
        start_token = None
        end_token = None

        # Encontra os índices dos tokens que correspondem ao span de caracteres
        for i, (t_start, t_end) in enumerate(token_spans):
            # Se o token estiver completamente dentro do span da entidade
            if t_start >= start_char and t_end <= end_char:
                if start_token is None:
                    start_token = i
                end_token = i
            # Se o token se sobrepõe ao span da entidade (parcialmente)
            elif (t_start < end_char and t_end > start_char):
                if start_token is None:
                    start_token = i
                end_token = i
        if start_token is not None and end_token is not None:
            ner.append([start_token, end_token, label])
    return {"tokenized_text": tokens, "ner": ner}

def load_dataset(path):
    """Carrega o dataset a partir de um arquivo JSONL."""
    with open(path, 'r', encoding='utf-8') as f:
        return [process_sample(json.loads(line)) for line in f]

def split_long_sentences(dataset, max_length=384, overlap=50):
    """
    Divide sentenças longas em segmentos menores com sobreposição (overlap).
    A sobreposição ajuda a preservar o contexto para entidades que caem nas fronteiras.
    """
    split_data = []
    for sample in dataset:
        words = sample.get("tokenized_text", [])
        ner_annotations = sample.get("ner", [])

        if not words or not isinstance(ner_annotations, list):
            continue

        if len(words) > max_length:
            # Calcula o passo para a divisão, considerando a sobreposição
            step = max_length - overlap
            if step <= 0: # Garante que o passo seja positivo
                step = max_length // 2 if max_length > 1 else 1

            for i in range(0, len(words), step):
                segment_start_idx = i
                segment_end_idx = min(i + max_length, len(words))

                # Ajusta o início do segmento para trás para incluir o overlap
                # Isso é importante para o primeiro segmento de cada "janela"
                if i > 0:
                    segment_start_idx = max(0, i - overlap)

                current_words = words[segment_start_idx:segment_end_idx]

                # Ajusta os spans das entidades para o novo segmento
                new_ner = []
                for start, end, label in ner_annotations:
                    # Verifica se a entidade está totalmente ou parcialmente dentro do segmento atual
                    if max(start, segment_start_idx) <= min(end, segment_end_idx - 1):
                        # Ajusta os índices de token da entidade para o novo segmento
                        adjusted_start = max(0, start - segment_start_idx)
                        adjusted_end = min(len(current_words) - 1, end - segment_start_idx)
                        # Apenas adiciona se a entidade ainda for válida após o ajuste
                        if adjusted_start <= adjusted_end:
                            new_ner.append([adjusted_start, adjusted_end, label])
                
                split_sample = {
                    "tokenized_text": current_words,
                    "ner": new_ner
                }
                # Adiciona a amostra dividida apenas se houver entidades anotadas nela
                if split_sample["ner"]:
                    split_data.append(split_sample)
        else:
            # Adiciona a amostra original se não for muito longa e tiver entidades
            if ner_annotations:
                split_data.append({"tokenized_text": words, "ner": ner_annotations})

    # Filtra novamente para garantir que todas as amostras tenham anotações NER válidas
    return [ex for ex in split_data if "ner" in ex and isinstance(ex["ner"], list) and ex["ner"]]


def create_dataloader(dataset, batch_size, collator, shuffle=True):
    """Cria um DataLoader para o dataset."""
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=shuffle)

def token_spans_to_char_offsets(tokens, spans):
    """
    Converte spans de tokens de volta para offsets de caracteres no texto original.
    Necessário para o cálculo do F1-score.
    """
    char_spans = []
    # Recria o texto original a partir dos tokens (assumindo espaço como delimitador)
    text = " ".join(tokens)
    current_pos = 0
    token_offsets = []

    # Calcula os offsets de caracteres para cada token
    for token in tokens:
        # Encontra a posição do token no texto recriado
        start = text.find(token, current_pos)
        end = start + len(token)
        token_offsets.append((start, end))
        current_pos = end + 1 # Move para depois do espaço

    # Converte os spans de tokens para spans de caracteres
    for start_idx, end_idx, label in spans:
        if start_idx < len(token_offsets) and end_idx < len(token_offsets):
            char_start = token_offsets[start_idx][0]
            char_end = token_offsets[end_idx][1]
            char_spans.append({"start": char_start, "end": char_end, "label": label})
    return text, char_spans

def f1_score_from_span_lists(pred_spans_list, gold_spans_list, average="macro"):
    """
    Calcula o F1-score comparando listas de spans de entidades preditas e gold.
    Considera um match exato de span (start, end, label).
    """
    true_labels = []
    pred_labels = []
    for pred_spans, gold_spans in zip(pred_spans_list, gold_spans_list):
        # Converte as listas de spans em conjuntos para facilitar a comparação
        gold_set = set((span["start"], span["end"], span["label"]) for span in gold_spans)
        pred_set = set((span["start"], span["end"], span["label"]) for span in pred_spans)
        
        # Cria um conjunto de todos os spans únicos (gold + preditos)
        all_spans = gold_set.union(pred_set)
        
        # Para cada span único, determina se ele estava no gold e/ou na predição
        for span in all_spans:
            true_labels.append(1 if span in gold_set else 0)
            pred_labels.append(1 if span in pred_set else 0)
    
    # Evita erro se true_labels ou pred_labels estiverem vazios
    if not true_labels and not pred_labels:
        return 1.0 # Perfeito se não há nada para prever e nada foi previsto
    if not true_labels or not pred_labels:
        return 0.0 # Se um está vazio e o outro não, é 0 F1

    return f1_score(true_labels, pred_labels, average=average, zero_division=0)

def compute_f1_by_threshold(model, dataset, threshold, entity_labels):
    """
    Calcula o F1-score para um dado modelo e dataset, aplicando um threshold de confiança.
    """
    all_preds = []
    gold_spans = [spans for _, spans in dataset] # Extrai apenas os spans gold

    for text, _ in dataset:
        # Realiza a predição de entidades com o modelo GLiNER
        preds = model.predict_entities(text, labels=entity_labels, threshold=threshold)
        # Filtra as predições para incluir apenas os rótulos de entidade de interesse
        filtered = [p for p in preds if p["label"] in entity_labels]
        all_preds.append(filtered)

    return f1_score_from_span_lists(all_preds, gold_spans, average="macro")

def clear_cuda_cache():
    """Limpa o cache da GPU para liberar memória."""
    gc.collect()
    torch.cuda.empty_cache()

# ==================== Configurações Principais =====================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_path = "Corpus_Grande_Pseudolabel_Threshold06.json"

# Carrega e pré-processa os dados
raw_data = load_dataset(train_path)
# Filtra amostras que não possuem anotações NER após o processamento inicial
filtered_data = [ex for ex in raw_data if ex["ner"]]
# Divide sentenças longas com sobreposição
dataset = split_long_sentences(filtered_data, max_length=384, overlap=100) # Aumentado o overlap

batch_size = 4
num_epochs = 20
n_splits = 3 # Número de folds para a validação cruzada externa
thresholds = [0.6] # Limiares para otimização do F1-score
model_base = "best_overall_gliner_model" # Modelo base GLiNER
# Extrai todos os rótulos de entidade únicos do dataset
entity_labels = list(sorted(list({label for sample in dataset for _, _, label in sample["ner"]})))

kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
best_overall_model = None
best_overall_score = 0.0
results_file = "resultados_nested_folds.txt"

# Abre o arquivo de resultados para escrita
with open(results_file, "w", encoding="utf-8") as f:
    f.write("Resultados Nested Cross-Validation\n\n")

# Loop de validação cruzada aninhada
for fold, (trainval_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"\n===== Fold Externo {fold+1}/{n_splits} =====")
    trainval_data = [dataset[i] for i in trainval_idx]
    test_data = [dataset[i] for i in test_idx]
    
    # Divisão interna para treinamento e validação (para otimização de hiperparâmetros)
    inner_train, inner_val = train_test_split(trainval_data, test_size=0.33, random_state=SEED)

    best_score_inner_fold = 0.0
    best_model_inner_fold = None
    best_thresh_inner_fold = 0.5
    best_params_inner_fold = (None, None)

    # Simulação de busca Bayesiana (random search com faixas ajustadas)
    # Para uma busca Bayesiana real com skopt.BayesSearchCV, seria necessário refatorar
    # a lógica de treinamento em um 'estimator' compatível com scikit-learn.
    # Aqui, mantemos a busca aleatória, mas com faixas de hiperparâmetros mais refinadas.
    for trial in range(5): # Número de tentativas de hiperparâmetros por fold interno
        # Faixa de Learning Rate (LR) ajustada: mais restrita e comum para fine-tuning
        # lr = round(float(np.random.uniform(5e-6, 5e-4)), 7) 
        lr = 3.38e-5
        # Faixa de Weight Decay (WD) ajustada: valores menores são mais comuns para regularização
        wd = 0.086619
        print(f"\n[Trial {trial+1}/5] LR={lr:.7f} | WD={wd:.6f}")

        model = GLiNER.from_pretrained(model_base)
        model.to(device)
        data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
        
        # O scheduler OneCycleLR é excelente para acelerar o treinamento
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=lr, 
            total_steps=num_epochs * len(inner_train) // batch_size + 1, 
            pct_start=0.1, 
            anneal_strategy="linear"
        )

        train_loader = create_dataloader(inner_train, batch_size, data_collator, shuffle=True)
        val_loader = create_dataloader(inner_val, batch_size, data_collator, shuffle=False)

        training_losses = []
        validation_losses = []
        best_epoch_score = -1.0 # Inicializa com um valor baixo para garantir que o primeiro F1 seja melhor
        patience = 7 # Aumentada a paciência para early stopping
        patience_counter = 0

        # Prepara os dados de validação para o cálculo do F1-score (char offsets)
        val_processed = [token_spans_to_char_offsets(ex["tokenized_text"], ex["ner"]) for ex in inner_val]

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                running_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clipping de gradientes
                optimizer.step()
                scheduler.step() # Atualiza o learning rate
                optimizer.zero_grad()
            epoch_train_loss = running_loss / len(train_loader)
            training_losses.append(epoch_train_loss)

            # Validação
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} (Val)", leave=False):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()
            epoch_val_loss = val_loss / len(val_loader)
            validation_losses.append(epoch_val_loss)

            # Cálculo do F1-score na validação para early stopping
            # Usamos um threshold fixo de 0.5 para o early stopping para simplificar
            f1_macro = compute_f1_by_threshold(model, val_processed, threshold=0.5, entity_labels=entity_labels)
            print(f"Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f} | Val Loss={epoch_val_loss:.4f} | F1-macro (Val)={f1_macro:.4f} | Patience ={patience_counter+1}/{patience}")

            # Lógica de Early Stopping baseada no F1-score
            if f1_macro > best_epoch_score:
                best_epoch_score = f1_macro
                patience_counter = 0
                # Opcional: Salvar o modelo nesta época se for o melhor até agora
                # best_model_current_trial = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping ativado na Época {epoch+1} por F1-score.")
                    break # Sai do loop de épocas

        # Avaliação final do modelo treinado no trial, otimizando o threshold
        # Re-calcula F1 para cada threshold para encontrar o melhor para este trial
        for thresh in thresholds:
            f1_macro_trial = compute_f1_by_threshold(model, val_processed, thresh, entity_labels)
            if f1_macro_trial > best_score_inner_fold:
                best_score_inner_fold = f1_macro_trial
                best_thresh_inner_fold = thresh
                # Salva uma cópia profunda do modelo se ele for o melhor no fold interno
                best_model_inner_fold = copy.deepcopy(model)
                best_params_inner_fold = (lr, wd)

        # Gráfico de perda para este trial
        plt.figure(figsize=(10, 6))
        plt.plot(training_losses, label="Train Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.title(f"Fold {fold+1} - Trial {trial+1} - LR={lr:.7f}, WD={wd:.6f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"loss_fold{fold+1}_trial{trial+1}.png")
        plt.close()

        clear_cuda_cache() # Limpa o cache da GPU após cada trial

    print(f"\nMelhores parâmetros para o Fold Interno {fold+1}: LR={best_params_inner_fold[0]:.7f} | WD={best_params_inner_fold[1]:.6f} | THRESH={best_thresh_inner_fold} | F1 (Val)={best_score_inner_fold:.4f}")

    # Avaliação no conjunto de teste externo com o melhor modelo e threshold encontrados
    test_processed = [token_spans_to_char_offsets(ex["tokenized_text"], ex["ner"]) for ex in test_data]
    
    # Verifica se best_model_inner_fold não é None antes de usar
    if best_model_inner_fold is not None:
        final_f1_test = compute_f1_by_threshold(best_model_inner_fold, test_processed, best_thresh_inner_fold, entity_labels)
        print(f"F1 final no teste do Fold Externo {fold+1}: {final_f1_test:.4f}")

        # Salva os resultados no arquivo
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(f"Fold Externo {fold+1}\n")
            f.write(f"Melhores parâmetros (LR, WD): {best_params_inner_fold[0]:.7f}, {best_params_inner_fold[1]:.6f}\n")
            f.write(f"Threshold escolhido: {best_thresh_inner_fold}\n")
            f.write(f"F1 no teste: {final_f1_test:.4f}\n")
            f.write("-" * 40 + "\n")

        # Atualiza o melhor modelo geral
        if final_f1_test > best_overall_score:
            best_overall_score = final_f1_test
            best_overall_model = copy.deepcopy(best_model_inner_fold)
            best_overall_model.save_pretrained("best_overall_gliner_model")
            print(f"Novo melhor modelo geral salvo com F1: {best_overall_score:.4f}")
    else:
        print(f"Nenhum modelo foi considerado 'melhor' no Fold Externo {fold+1}. Verifique os dados de validação.")

print(f"\nTreinamento concluído. Melhor F1-score geral no teste: {best_overall_score:.4f}")
if best_overall_model:
    print("O melhor modelo geral foi salvo em 'best_overall_gliner_model'.")
else:
    print("Nenhum modelo geral foi salvo. Verifique o processo de treinamento.")