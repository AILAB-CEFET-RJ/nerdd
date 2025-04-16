import csv

def load_word_label_pairs(filepath):
    """Carrega apenas os pares (word, label) do train_data."""
    pairs = set()
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pair = (row['word'], row['label'])
            pairs.add(pair)
    return pairs

def find_differences_with_ids(train_file, test_file):
    """Compara os arquivos e retorna as linhas de test_data que têm (word, label) ausentes no train_data."""
    train_pairs = load_word_label_pairs(train_file)
    differences = []

    with open(test_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pair = (row['word'], row['label'])
            if pair not in train_pairs:
                differences.append((row['sentence_id'], row['word'], row['label']))

    return differences

def save_differences_to_csv(differences, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sentence_id', 'word', 'label'])  # Cabeçalho
        for sent_id, word, label in differences:
            writer.writerow([sent_id, word, label])

# Uso
train_file = 'Bases/train_data.csv'
test_file = 'Bases/test_data.csv'
output_file = 'sanity_test.csv'

diffs = find_differences_with_ids(train_file, test_file)
save_differences_to_csv(diffs, output_file)

print(f"{len(diffs)} diferenças salvas em '{output_file}'")
