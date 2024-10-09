import csv
import json

caminho_arquivo_csv = "Bases/test_data.csv"

def process_csv_to_json(file_path):
    tokenized_sentences = []
    current_sentence = []
    ner_annotations = []
    current_ner = []

    # Abre o arquivo CSV e processa linha a linha
    with open(file_path, 'r', encoding='utf-8') as file:
        # Lê todo o conteúdo do arquivo
        content = file.read()

        #sentences = content.replace("\"\n\"", "@@")

        # Divide o conteúdo em sentenças usando '@@' como delimitador
        sentences = content.split("\"\n\"")

        # Processa cada sentença separada
        for sentence in sentences:
            # Divide a sentença em linhas por quebra de linha
            lines = sentence.strip().split("\n")

            # Processa cada linha dentro da sentença
            for line in lines:
                line = line.strip()  # Remove espaços em branco extras e quebras de linha

                # Verifica se a linha contém ponto e vírgula
                if ";" not in line:
                    continue  # Ignora linhas mal formatadas ou em branco

                # Tenta separar o token da entidade, verificando se há exatamente dois elementos
                parts = line.split(";")

                if len(parts) != 2:
                    continue  # Ignora linhas que não têm exatamente dois elementos (token e entidade)

                token, entity = parts
                token = token.strip()
                entity = entity.strip()

                # Adiciona o token à sentença atual
                current_sentence.append(token)

                # Se o token tiver uma entidade (diferente de O), anota sua posição
                if entity != "O":
                    current_ner.append([len(current_sentence)-1, len(current_sentence)-1, entity.lower()])

            # Adiciona a sentença ao JSON após processar todas as linhas dentro dela
            if current_sentence:
                tokenized_sentences.append({
                    "tokenized_text": current_sentence,
                    "ner": current_ner
                })
                current_sentence = []
                current_ner = []

    # Retorna o JSON com as sentenças tokenizadas e as entidades
    return json.dumps(tokenized_sentences, ensure_ascii=False, indent=2)

# Converte o CSV para JSON
json_result = process_csv_to_json(caminho_arquivo_csv)
output_json_path = "resultado-teste.json"
# Salva a string JSON em um arquivo
with open(output_json_path, 'w', encoding='utf-8') as json_file:
  json_file.write(json_result)