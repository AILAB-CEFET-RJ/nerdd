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
import json
from transformers import AutoTokenizer
import pandas as pd
import unicodedata
from sklearn.model_selection import train_test_split

INPUT_FILE = "Corpus_grande_sample.json"
OUTPUT_FILE = "Corpus_grande_pseudolabel_score0.json"


def ler_modelo(model_base):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GLiNER.from_pretrained(model_base, load_tokenizer=True)
    model.to(device)
    tokenizer = model.data_processor.transformer_tokenizer
    tokenizer.model_max_length = 1024
    return model, tokenizer

def ler_arquivo(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)

def split_by_token_limit(text, max_tokens=384, tokenizer=None):
    if tokenizer is None:
        raise ValueError("Tokenizer não foi fornecido.")
    
    tokens = tokenizer(text, return_offsets_mapping=True, truncation=False)
    input_ids = tokens['input_ids']
    offsets = tokens['offset_mapping']

    chunks = []
    start = 0
    while start < len(input_ids):
        end = min(start + max_tokens, len(input_ids))
        chunk_ids = input_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunk_offset = offsets[start][0]
        chunks.append((chunk_text, chunk_offset))
        start = end

    return chunks


def predict_modelo_relato(modelo, sample,tokenizer):
    """
    Função para prever entidades apenas no campo 'relato' com o modelo GLiNER.
    """
    relato = sample.get("relato", "")
    all_entities = []

    try:
        chunks = split_by_token_limit(relato, 384,tokenizer)
        for chunk_text, offset in chunks:
            predicted_entities = modelo.predict_entities(
                chunk_text, 
                labels=['Person', 'Organization', 'Location']
            )
            for ent in predicted_entities:
                if ent.get("score", 1.0) >= 0:
                    ent["start"] += offset
                    ent["end"] += offset
                    all_entities.append(ent)
    except Exception as e:
        print(f"Erro ao processar relato: {relato[:30]}... — {str(e)}")

    output_sample = {
        "assunto": sample.get("assunto", ""),
        "relato": relato,
        "logradouroLocal": sample.get("logradouroLocal", ""),
        "bairroLocal": sample.get("bairroLocal", ""),
        "cidadeLocal": sample.get("cidadeLocal", ""),
        "pontodeReferenciaLocal": sample.get("pontodeReferenciaLocal", ""),
        "entities": all_entities
    }

    return output_sample

# Função para normalizar texto
def normalizar(texto):
    if not texto:
        return ""
    texto = unicodedata.normalize('NFKD', texto)
    texto = texto.encode('ASCII', 'ignore').decode('utf-8')
    return texto.lower().strip()

def ajustar_metadados(item):
    # Ajusta scores das entidades de uma única linha com base em localização
    referencias = {
        normalizar(item.get('bairroLocal')),
        normalizar(item.get('logradouroLocal')),
        normalizar(item.get('cidadeLocal')),
        normalizar(item.get('pontodeReferenciaLocal'))
    }

    if 'entities' in item and item['entities']:
        for entidade in item['entities']:
            texto_normalizado = normalizar(entidade['text'])
            score_antigo = entidade['score']
            if texto_normalizado in referencias:
                score_novo = min(score_antigo * 1.2, 1.0)
                entidade['score'] = score_novo
    return item

def limiar_confianca_df(linha, limiar_confianca):
    entidades_filtradas = [ent for ent in linha['entities'] if ent['score'] > limiar_confianca]
    if entidades_filtradas:
        linha_filtrada = linha.copy()
        linha_filtrada['entities'] = entidades_filtradas
        return linha_filtrada
    return None

def ajustar_exemplo_para_gliner(ex):
    return {
        "text": ex["relato"],
        "ner": [
            {
                "start": int(ent["start"]),
                "end": int(ent["end"]),
                "label": ent["label"]
            }
            for ent in ex.get("entities", [])
        ]
    }

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

def create_dataloader(dataset, batch_size, collator, shuffle=True):
    """Cria um DataLoader para o dataset."""
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=shuffle)

def token_spans_to_char_offsets(text, entities):
    token_offsets = [...]  # sua lógica de obtenção dos offsets
    new_entities = []

    for ent in entities:
        start_idx = int(ent["start"])  # conversão aqui
        end_idx = int(ent["end"])      # conversão aqui

        if start_idx < len(token_offsets) and end_idx < len(token_offsets):
            char_start = token_offsets[start_idx][0]
            char_end = token_offsets[end_idx][1]
            new_entities.append({
                "start": char_start,
                "end": char_end,
                "label": ent["label"]
            })

    return {"text": text, "spans": new_entities}


def ajuste_fino_df(modelo_inicial, corpus_pseudo, num_epochs=3, patience=1, batch_size=16, lr=1e-5, wd=0.01, thresholds=0.5):
    """
    Realiza ajuste fino do modelo usando uma divisão treino/teste dos dados pseudo-rotulados.
    """
    if not corpus_pseudo:
        print("Nenhum dado pseudo-rotulado disponível.")
        return None

    # Divide os dados em treino e validação
    train_data, val_data = train_test_split(corpus_pseudo, test_size=0.33, random_state=42)
    entity_labels = ['Person', 'Organization', 'Location']

    # Carrega modelo e tokenizer
    model, tokenizer = ler_modelo(modelo_inicial)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Prepara collator
    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

    # Cria os dataloaders
    # train_data = [item[1] for item in train_data if item[1] is not None]
    train_data = [ajustar_exemplo_para_gliner(ex[1]) for ex in corpus_pseudo.iterrows()]
    val_processed = [token_spans_to_char_offsets(ex[1]["relato"],[{
                "start": int(ent["start"]),
                "end": int(ent["end"]),
                "label": ent["label"]
            }
            for ent in ex[1]["entities"]
        ]
    )
    for ex in val_data]

    if not train_data:
        print("Dados de treino vazios após filtragem.")
        return model

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = create_dataloader(val_data, batch_size, data_collator, shuffle=False)

    # Otimizador
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    training_losses = []
    validation_losses = []
    best_epoch_score = 0.0
    patience_counter = 0

    # Loop de treinamento
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            running_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
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

        # F1 para early stopping
        f1_macro = compute_f1_by_threshold(model, val_processed, threshold=thresholds, entity_labels=entity_labels)
        print(f"Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f} | Val Loss={epoch_val_loss:.4f} | F1-macro (Val)={f1_macro:.4f} | Patience={patience_counter+1}/{patience}")

        if f1_macro > best_epoch_score:
            best_epoch_score = f1_macro
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping ativado na Época {epoch+1} por F1-score.")
                break

    return model


def auto_treinamento_iterativo(modelo_inicial,corpus_nao_rotulado,num_iteracoes,
                               limiar_confianca,ajuste_fino):
    modelo, tokenizer = ler_modelo(modelo_inicial)
    DD_nao_rot = ler_arquivo(corpus_nao_rotulado)
    
    for t in range(num_iteracoes):
        corpus_pseudo = []

        for idx, row in DD_nao_rot.iterrows():
            output_predict = predict_modelo_relato(modelo,row,tokenizer)
            df_metadados = ajustar_metadados(output_predict)
            linha_filtrada = limiar_confianca_df(df_metadados, limiar_confianca)
            
            corpus_pseudo.append((row, linha_filtrada))

        # Ajuste fino com os dados pseudo-rotulados
        print(corpus_pseudo[0])
        #print(type(corpus_pseudo))
        modelo = ajuste_fino_df(modelo_inicial, corpus_pseudo,num_epochs=5, patience=1, batch_size=16, lr=3.38e-5, wd=0.086619,thresholds=limiar_confianca)
        modelo.save_pretrained(f"gliner_model_iter_{t+1}")
        
        # Remove os exemplos pseudo-rotulados do conjunto não rotulado
        DD_nao_rot = [x for x in DD_nao_rot if x not in {item[0] for item in corpus_nao_rotulado}]

    return modelo


corpus_pseudo = auto_treinamento_iterativo(
    modelo_inicial="best_overall_gliner_model",
    corpus_nao_rotulado=INPUT_FILE,
    num_iteracoes=1,
    limiar_confianca=0.6,
    ajuste_fino=None  # Passe uma função de ajuste fino se necessário
)
