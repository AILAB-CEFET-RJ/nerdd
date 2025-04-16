import csv
import unicodedata

# Function to read CSV file and return a set of words
def ler_csv(arquivo):
    with open(arquivo, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        return {linha[0] for linha in reader}

# Function to normalize words
def normalizar_palavra(palavra):
    # Convert to lowercase
    palavra = palavra.lower()
    # Remove accents and special characters
    palavra = ''.join(
        c for c in unicodedata.normalize('NFD', palavra)
        if unicodedata.category(c) != 'Mn'
    )
    # Remove extra spaces
    palavra = palavra.strip()
    return palavra

# Read and normalize words from the provided CSV files
with open('Bases/entities_output_TOTAL.csv', 'r', encoding='utf-8') as f:
    palavras_arquivo1 = {normalizar_palavra(row[0]) for row in csv.reader(f)}

with open('Bases/test_data.csv', 'r', encoding='utf-8') as f:
    palavras_arquivo2 = {normalizar_palavra(row[0]) for row in csv.reader(f)}

# Identify missing words in each file
faltantes_arquivo1 = palavras_arquivo2 - palavras_arquivo1
faltantes_arquivo2 = palavras_arquivo1 - palavras_arquivo2

# Save missing words to separate CSV files
with open('faltantes_no_entities_output.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Palavras Faltantes no entities_output.csv'])
    for palavra in faltantes_arquivo1:
        writer.writerow([palavra])

with open('faltantes_no_test_data_sentence.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Palavras Faltantes no test_data_sentence.csv'])
    for palavra in faltantes_arquivo2:
        writer.writerow([palavra])

print("As palavras faltantes foram salvas nos arquivos 'faltantes_no_entities_output.csv' e 'faltantes_no_test_data_sentence.csv'.")
