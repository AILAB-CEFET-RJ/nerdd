# NERDD  - Named Entity Recognition Disque Denúncia 🚫

**NERDD** is a tool that utilizes the pretrained **Word2vec**,**LSTM** and **Spacy**  model to detect the entity in the violence text. Additionally, it employs another pretrained model (which will be mentioned later) to perform classification and check for NER , specifically identify local, person and time or organization.

Este projeto é uma maneira divertida e interessante de explorar de detecção de entidades em textos de violencia. Vamos ajudá-lo a configurar e executar o NERDD em seu ambiente.

## Prerequisites 📋

Make sure you have the following prerequisites installed in your development environment:

- Python 3.x
- Pip (Python package manager)
- Spacy
- Word2vec
- LSTM

## How to Run 🏃‍♀️

Follow these simple steps to set up and run HANDY:

1. **Clone the Repository**

   ```shell
   git clone https://github.com/MLRG-CEFET-RJ/nerdd.git
   ```

2. **Install Dependencies**

   ```shell
   cd lstm_ner
   pip install -r requirements.txt
   ```

3. **Run HANDY**

   ```shell
   python lstm_ner/__main__.py
   ```

   Now you should see the live camera window with hand detection and offensive gesture classification.


## Contribution 🤝

If you want to contribute to the NERDD project, we would be happy to receive your contributions. Feel free to open issues or submit pull requests with improvements, bug fixes, or new features.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

We hope you enjoy offensive gesture detection with **NERDD**! If you have any questions or need assistance, please feel free to reach out to the development team. 😊👋
