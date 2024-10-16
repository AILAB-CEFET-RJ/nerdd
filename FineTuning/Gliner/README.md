# NERDD  - FineTuning - GliNER 🚫

**NERDD - GliNER** is a Named Entity Recognition (NER) model capable of identifying any type of entity using a bidirectional transformer encoder (BERT-like). It provides a practical alternative to traditional NER models, which are limited to predefined entities, and Large Language Models (LLMs) which, despite their flexibility, are expensive and large for resource-constrained scenarios.

In this example, GliNER is used to identify entities in the text that specify the entity of violence. In addition, it employs another pre-trained model (which will be mentioned later) to perform classification and verify the NER, specifically identifying the location, person, and time or organization.

The GliNER documentation and thesis are described in the links below:

- https://github.com/urchade/GLiNER
- https://arxiv.org/abs/2311.08526

The files are arranged in GliNER for the creation of the JSON file for FineTuning (GliNER - JSON Criacao.py), the file in which the GliNER model is generated, the output file with the generated tests (GliNER.py) and the result of the GliNER file compared to the files generated for the tests (GliNER - Resultado.py).

The steps to run GliNER:

    1 - Run the file "GliNER - Criacao JSON.py" with the python command pointing to a CSV file with the following markup (token;markup)
    2 - Then run the file "GliNER.py" to generate the training and data markup through GliNER. (Note: This file must be run on a GPU with at least 16 GB)
    3 - Run the file "GliNER - Resultado.py" which, at the end of the execution, will generate a report with the classification metrics.

----------------------------------------------------------------------------------------
**NERDD - GliNER** é um modelo Named Entity Recognition (NER) capaz de identificar qualquer tipo de entidade usando um codificador transformador bidirecional (tipo BERT). Ele fornece uma alternativa prática aos modelos NER tradicionais, que são limitados a entidades predefinidas, e Large Language Models (LLMs) que, apesar de sua flexibilidade, são caros e grandes para cenários com recursos limitados.
Neste exemplo o GliNER é utilizado para identificar entidades no texto que especificam a entidades de violencia.Além disso, emprega outro modelo pré-treinado (que será mencionado posteriormente) para realizar a classificação e verificar o NER, identificando especificamente o local, a pessoa e o horário ou organização.

As documentações e a tese do GliNER estão descritos nos links abaixo:

- https://github.com/urchade/GLiNER
- https://arxiv.org/abs/2311.08526


Os arquivos estão dispostos no GliNER para a criação do arquivo JSON para o FineTuning (GliNER - Criacao JSON.py), o arquivo em que gera o modelo do GliNER, o arquivo de saída com os testes gerados (GliNER.py) e o resultado do arquivo GliNER em comparado com os arquivos gerados para os testes (GliNER - Resultado.py).

Os passos para rodar o GliNER :

    1 - Executar o arquivo "GliNER - Criacao JSON.py" com o comando python apontando para um arquivo CSV com a seguinte marcação (token;marcação)
    2 - Em seguida executar o arquivo "GliNER.py" para gerar os treinos e a marcação dos dados através do GliNER. (Obs: Esse arquivo deve ser rodado em uma GPU com no minimo 16 gbs)
    3 - Deve se executar o arquivo "GliNER - Resultado.py" que ao final da execução irá gerar um relatório com as métricas das classificações. 

## Prerequisites 📋

Make sure you have the following prerequisites installed in your development environment:

- Python 3.x
- Pip (Python package manager)
- GliNER
- accelarate
- csv
- json

## How to Run 🏃‍♀️

Follow these simple steps to set up and run NERDD:

1. **Clone the Repository**

   ```shell
   git clone https://github.com/MLRG-CEFET-RJ/nerdd.git
   ```

## Contribution 🤝

If you want to contribute to the NERDD project, we would be happy to receive your contributions. Feel free to open issues or submit pull requests with improvements, bug fixes, or new features.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
