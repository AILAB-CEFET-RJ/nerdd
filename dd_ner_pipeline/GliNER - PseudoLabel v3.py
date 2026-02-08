import json
import os
import re
from tqdm import tqdm
import torch
from gliner import GLiNER
from transformers import AutoTokenizer

# === CONFIGURAÇÕES ===
MODEL_PATH = "models-adjust/best_overall_gliner_model"
INPUT_FILE = "Corpus_grande.json"
OUTPUT_FILE = "Corpus_grande_pseudolabel_score0.json"
MAX_TOKENS = 384

# === 1. Carregar modelo e tokenizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

print("Carregando modelo...")
model = GLiNER.from_pretrained(MODEL_PATH)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("Modelo carregado com sucesso.")

# === 2. Carregar textos ===
print(f"Carregando textos de {INPUT_FILE}...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

print(f"{len(raw_data)} amostras carregadas.")

# === 3. Função para dividir texto em janelas de até 384 tokens ===
def split_by_token_limit(text, max_tokens=384):
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    chunks = []
    start = 0

    while start < len(input_ids):
        end = min(start + max_tokens, len(input_ids))
        chunk_offsets = offsets[start:end]

        chunk_start_char = chunk_offsets[0][0]
        chunk_end_char = chunk_offsets[-1][1]
        chunk_text = text[chunk_start_char:chunk_end_char]

        chunks.append((chunk_text, chunk_start_char))
        start = end

    return chunks

# === 4. Gerar pseudo-rótulos ===
pseudo_labeled_data = []

print("Gerando pseudo-rótulos...")
for sample in tqdm(raw_data):
    # Campos originais
    assunto = sample.get("assunto", "")
    relato = sample.get("relato", "")
    bairro = sample.get("bairroLocal", "")
    logradouro = sample.get("logradouroLocal", "")
    cidade = sample.get("cidadeLocal", "")
    ponto_ref = sample.get("pontodeReferenciaLocal", "")

    # Texto para predição
    text_parts = [assunto, relato, bairro, logradouro, cidade, ponto_ref]
    full_text = ". ".join([p for p in text_parts if p]).strip()

    all_entities = []

    try:
        chunks = split_by_token_limit(full_text, MAX_TOKENS)

        for chunk_text, offset in chunks:
            predicted_entities = model.predict_entities(chunk_text, labels=['Person', 'Organization', 'Location'])
            for ent in predicted_entities:
                #if ent.get("score", 1.0) >= 0.6:
                if ent.get("score", 1.0) >= 0:
                    ent["start"] += offset
                    ent["end"] += offset
                    all_entities.append(ent)

    except Exception as e:
        print(f"Erro ao processar texto: {full_text[:30]}... — {str(e)}")

    # Montar saída final
    output_sample = {
        "assunto": assunto,
        "relato": relato,
        "logradouroLocal": logradouro,
        "bairroLocal": bairro,
        "cidadeLocal": cidade,
        "pontodeReferenciaLocal": ponto_ref,
        "entities": all_entities
    }

    pseudo_labeled_data.append(output_sample)

# === 5. Salvar resultados ===
print(f"Salvando pseudo-rótulos em {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for sample in pseudo_labeled_data:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print("✅ JSON gerado no formato correto com sucesso.")