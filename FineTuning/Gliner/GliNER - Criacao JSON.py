import csv
import json

caminho_arquivo_csv = "Bases/train_data.csv"

def csv_to_json_no_sentence_ids(file_path):
    # Lista para armazenar cada sentença como um item separado
    sentences = []
    
    # Dicionário temporário para a sentença atual
    current_sentence = None
    current_sentence_id = None

    # Lendo o CSV
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            sentence_id = int(row["sentence_id"])
            word = row["word"]
            label = row["label"]

            # Se encontramos um novo sentence_id, salvamos a sentença anterior e criamos uma nova
            if sentence_id != current_sentence_id:
                if current_sentence is not None:
                    sentences.append(current_sentence)
                current_sentence = {
                    "tokenized_text": [],
                    "ner": []
                }
                current_sentence_id = sentence_id

            # Adiciona a palavra ao tokenized_text da sentença atual
            index = len(current_sentence["tokenized_text"])
            current_sentence["tokenized_text"].append(word)
            
            # Adiciona cada palavra com um rótulo dinâmico ao ner
            if label:  # Checa se há um rótulo
                current_sentence["ner"].append([index, index, label.lower()])

        # Adiciona a última sentença ao final do loop
        if current_sentence is not None:
            sentences.append(current_sentence)

    # Salva todas as sentenças em um único arquivo JSON
    with open('resultado-treino.json', 'w', encoding='utf-8') as json_file:
        json.dump(sentences, json_file, ensure_ascii=False, indent=4)
    
    print("Arquivo JSON único gerado com sucesso!")




csv_to_json_no_sentence_ids(caminho_arquivo_csv)

# Salvar como JSON
#with open('output.json', 'w', encoding='utf-8') as json_file:
#    json.dump(output, json_file, ensure_ascii=False, indent=4)

print("Arquivo JSON gerado com sucesso!")