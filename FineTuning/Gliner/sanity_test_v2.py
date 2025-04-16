import csv

def load_word_label_pairs(filepath):
    """Carrega todos os pares (word, label) únicos do train_data."""
    pairs = set()
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pairs.add((row['word'], row['label']))
    return pairs

def group_sentences(filepath):
    """Agrupa palavras por sentence_id em ordem."""
    sentences = {}
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sid = int(row['sentence_id'])
            if sid not in sentences:
                sentences[sid] = []
            sentences[sid].append((row['word'], row['label']))
    return sentences

def find_full_entity_blocks(train_file, test_file):
    train_pairs = load_word_label_pairs(train_file)
    test_sentences = group_sentences(test_file)

    differences = []

    for sent_id, tokens in test_sentences.items():
        i = 0
        while i < len(tokens):
            word, label = tokens[i]
            if label.startswith("B-"):
                entity_type = label[2:]
                entity_block = [(sent_id, word, label)]
                j = i + 1
                while j < len(tokens):
                    next_word, next_label = tokens[j]
                    if next_label == f"I-{entity_type}":
                        entity_block.append((sent_id, next_word, next_label))
                        j += 1
                    else:
                        break

                # Se qualquer (word, label) da entidade não estiver no train_data → salva o bloco todo
                if any((w, l) not in train_pairs for (_, w, l) in entity_block):
                    differences.extend(entity_block)

                i = j  # pula o bloco inteiro
            else:
                i += 1

    return differences

def save_differences_to_csv(differences, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sentence_id', 'word', 'label'])  # Cabeçalho
        for row in differences:
            writer.writerow(row)

# ---------- USO ---------- #
train_file = 'Bases/train_data.csv'
test_file = 'Bases/test_data.csv'
output_file = 'diferencas.csv'

diffs = find_full_entity_blocks(train_file, test_file)
save_differences_to_csv(diffs, output_file)

print(f"{len(diffs)} tokens (entidades completas) salvos em '{output_file}'")
