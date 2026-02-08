import json
import csv
from nltk.tokenize import TreebankWordTokenizer

# Nome dos arquivos
input_file = "gliner_teste_sanidade.json"
output_file = "gliner_teste_sanidade_bio.csv"

# Inicializa o tokenizer
tokenizer = TreebankWordTokenizer()
sentence_id = 0

# Abre arquivos
with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:

    writer = csv.writer(outfile)
    writer.writerow(["sentence_id", "word", "label"])

    for line in infile:
        if not line.strip():
            continue  # pular linhas vazias

        data = json.loads(line)
        text = data["text"]
        spans = data.get("spans", [])

        # Tokenizar texto
        tokens = tokenizer.tokenize(text)
        token_spans = list(tokenizer.span_tokenize(text))

        for token, (start, end) in zip(tokens, token_spans):
            label = "O"
            for span in spans:
                if start >= span["start"] and end <= span["end"]:
                    tag = "B" if start == span["start"] else "I"
                    ent_type = span["label"].upper()[:3]  # e.g., "PER", "LOC"
                    label = f"{tag}-{ent_type}"
                    break
            writer.writerow([sentence_id, token, label])
        sentence_id += 1

print(f"Conversão concluída: {output_file}")
