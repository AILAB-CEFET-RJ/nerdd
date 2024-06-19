# MINERAÇÃO DE TEXTO - ✒️✏️

**Text Mining** was an article made to compare the three models for recognizing entities in three different libraries: Spacy, NLTK and ChatGPT API. This project is divided into two parts the Notebooks folder with four ipynb files and the streamlit folder to run the application that identifies the recognition entity according to what is written by the user.

In the Notebooks folder, there are four files, one with the NLTK model, the "Spacy - Artigo" file is the test without obtaining pre-training with the file while "Spacy - Modelos" has pre-training done manually. The "NLTK -Modelos" has the training done by NLTK while "Testes-ChatGPT" are the tests done by the ChatGPT API.

In the Streamlit folder we have the application, however for it to become functional we have to run the notebooks above that will generate the available h5 models. This folder contains the chatgpt.py file which has the connection with ChatGPT for extracting the recognition entities.

This project is a way to compare models for entity recognition extraction and compare these models to explore NER detection. We will help you configure and run Text Mining in your environment.

------------------------------------------------------------------------------

Mineração de Texto foi um artigo feito para comparar os tres modelos para o reconhecimento de entidades em tres bibliotecas diferentes : Spacy, NLTK e API do ChatGPT.Este projeto é dividido em duas partes a pasta Notebooks com quatro arquivos ipynb e a pasta streamlit para rodar a aplicação que identifica a entidade de reconhecimento de acordo com o que for escrito pelo usuário.

Na pasta  Notebooks, são quatro arquivos um com o modelo NLTK, o arquivo "Spacy - Artigo" é o teste sem obter um pré-treinamento com o arquivo enquanto o  "Spacy - Modelos" possui um pré-treinamento feito manualmente.O "NLTK-Modelos" tem o treinamento feito pelo NLTK enquanto o "Testes-ChatGPT" são os testes feito pela API do ChatGPT. 

Na pasta Streamlit temos a aplicação entretanto para ela se tornar funcional temos que rodar os notebooks acima que irão gerar os modelos h5 disponiveis, essa pasta contem o arquivo chatgpt.py que possui a conexão com o ChatGPT para a extração das entidades de reconhecimento.

Este projeto é uma maneira de comparar os modelos para a extração de reconhecimento de entidade e comparar esses modelos para explorar a detecção das NER.Vamos ajudá-lo a configurar e executar o Mineração de Texto em seu ambiente.

## Prerequisites 📋

Make sure you have the following prerequisites installed in your development environment:

- Python 3.x
- Pip (Python package manager)
- Streamlit
- NLTK
- Spacy
- API ChatGPT

## How to Run 🏃‍♀️

Follow these simple steps to set up and run Text Mining:

1. **Clone the Repository**

   ```shell
   git clone https://github.com/AIRGOLAB-CEFET-RJ/nerdd.git
   ```

2. **Install Dependencies**

   ```shell
   cd Mineração de Texto
   pip install -r requirements.txt
   ```

3. **Run Streamlit**

   ```shell
   streamlit run index.py
   ```

## Results Codes 🧪

In the `resultados/` directory, you will find some results to test Text-Mining functionality. To run the tests, you can use the following command:

This will run the offensive gesture detector on the test images and videos and display the results in the output.

## Contribution 🤝

If you want to contribute to the Text Mining project, we would be happy to receive your contributions. Feel free to open issues or submit pull requests with improvements, bug fixes, or new features.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

We hope you enjoy offensive gesture detection with **TEXT MINING**! If you have any questions or need assistance, please feel free to reach out to the development team. 😊👋
