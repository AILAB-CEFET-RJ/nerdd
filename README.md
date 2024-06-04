# NERDD  - Named Entity Recognition Disque Denúncia 🚫

**NERDD** is a tool that utilizes the pretrained **Word2vec**,**LSTM**, **Spacy** and **LLM**  model to detect the entity in the violence text. Additionally, it employs another pretrained model (which will be mentioned later) to perform classification and check for NER , specifically identify local, person and time or organization.

This repository is divided into four parts:
- **Text Mining** which was a project developed for the matter where we have 1000 criminal reports and are compared between **Spacy**, **Chat GPT** and **NLTK** as well as a tool using Streamlit to identify the entities of recognition automatically.
- **Word2vec - Leonidia** is the project based on student Leonidia's monograph using Word Embeddings to classify and identify recognition entities
- **Lstm_ner** is the project based on student Leonidia's monograph using Word Embeddings, however it uses EarlyStopping to generate a graph with a plot of the information generated in each season
- **Fine Tuning** are experiments using LLM to use SFT (Supervised Fine Tuning) to be used later in the Disque Denúncia database

This project is an interesting way to explore entity detection in violent texts. We will help you configure and run NERDD in your environment.

----------------------------------------------------------------------------------------
**NERDD** é uma ferramenta que utiliza os modelos pré-treinados **Word2vec**,**LSTM**, **Spacy** e **LLM** para detectar a entidade no texto de violência. Além disso, emprega outro modelo pré-treinado (que será mencionado posteriormente) para realizar a classificação e verificar o NER, identificando especificamente o local, a pessoa e o horário ou organização.

Este repositório encontra-se dividido em quatro partes:

- **Mineração de Texto** que foi um projeto desenvolvido para a matéria onde temos 1000 relatos criminais e estão comparados entre **Spacy**,**Chat GPT** e **NLTK** além de uma ferramenta utilizando o Streamlit para identificar as entidades de reconhecimento automaticamente.
- **Word2vec - Leonidia** é o projeto baseado na monografia da aluna Leonidia utilizando Word Embeddings para fazer a classificação e identificação das entidades de reconhecimentos
- **Lstm_ner** é o projeto baseado na monografia da aluna Leonidia utilizando Word Embeddings entretanto utiliza EarlyStopping para gerar um gráfico com plot das informações gerada em cada época
- **Fine Tuning** são experimentos utilizando o LLM para utilizar o SFT (Supervisioned Fine Tuning) para ser utilizado posteriormente na base de dados do Disque Denúncia
Este projeto é uma maneira interessante de explorar de detecção de entidades em textos de violencia. Vamos ajudá-lo a configurar e executar o NERDD em seu ambiente.

## Prerequisites 📋

Make sure you have the following prerequisites installed in your development environment:

- Python 3.x
- Pip (Python package manager)
- Spacy
- Word2vec
- LSTM
- Pytorch

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

We hope you enjoy offensive gesture detection with **NERDD**! If you have any questions or need assistance, please feel free to reach out to the development team. 😊👋