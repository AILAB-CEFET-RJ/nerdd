# NERDD  - FineTuning 🚫

**NERDD - FineTuning** is a tool that utilizes the pretrained  **LLM**  model to detect the entity in the violence text. Additionally, it employs another pretrained model (which will be mentioned later) to perform classification and check for NER , specifically identify local, person and organization.

- **Fine Tuning** are experiments using LLM to use SFT (Supervised Fine Tuning),BERT, UBIAI and principally GliNER to be used later in the Disque Denúncia database

This project is an interesting way to explore entity detection in violent texts. We will help you configure and run NERDD in your environment.

----------------------------------------------------------------------------------------
**NERDD** é uma ferramenta que utiliza os modelos pré-treinados **Word2vec**,**LSTM**, **Spacy** e **LLM** para detectar a entidade no texto de violência. Além disso, emprega outro modelo pré-treinado (que será mencionado posteriormente) para realizar a classificação e verificar o NER, identificando especificamente o local, a pessoa e o horário ou organização.

Este repositório encontra-se dividido em quatro partes:


- **Fine Tuning** são experimentos utilizando o LLM para utilizar o SFT (Supervisioned Fine Tuning),BERT, UBIAI e principalmente GliNER para ser utilizado posteriormente na base de dados do Disque Denúncia
Este projeto é uma maneira interessante de explorar de detecção de entidades em textos de violencia. Vamos ajudá-lo a configurar e executar o NERDD em seu ambiente.

## Prerequisites 📋

Make sure you have the following prerequisites installed in your development environment:

- Python 3.x
- Pip (Python package manager)
- Spacy
- Word2vec
- LSTM
- Pytorch
- GliNER

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
