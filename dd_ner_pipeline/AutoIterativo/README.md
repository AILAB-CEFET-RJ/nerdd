# GLINERDD  -  AutoIterative 🚫

**GLINERDD - AutoIterative** are the functional scripts present in the articles and using the algorithm identified below:  

```shell
   M = Initial_Model
   DD unlabeled = DD large
   for t=1 to T do
      DD pseudo = 0
      for all x in DD unlabeled do
         y,confidence = Model(x)
         C_calibrated = ModelCalibration(c, M_calibrated)
         if MetadataConfirms(y,x) then
            c = min(C_calibrated*confidence_factor,1)
         end if 
         if confidence_factor >= confidence_threshold then
            DDpseudo = DDpseudo(x,y)
         end if
      end for
      M = FineTune(M,DDpseudo)
      DD unlabeled = DD unlabeled \ DDpseudo
   end for
   return M
```
To run the iterative algorithm process, use the script below:


```shell
python main_iterativo.py \
  --input Corpus_grande.json \
  --csv   comparacao_calibracao.csv \
  --iter  5 \
  --tau   0.80 \
  --alpha 1.2 \
  --model modelo_inicial \
  --out   saida_iterativa
```

The parameters used to run the process were:

- Input – File to be read
- CSV – File used for calibration
- Iter – Number of generated iterations
- tau – Confidence threshold value
- alpha – Metadata confirmation increment value
- Model – The initial model used
- out – The folder containing the output with results and models

Running this process requires computational resources; therefore, there is a folder named "separado" (separate) to avoid unnecessary computational costs.


--------------------------------------------------------------------------------------------------------

**GLINERDD - AutoIterativo** são os scripts fuuncionais presentes nos artigos e utilizando o algoritmo presente que está identificado abaixo:  

```shell
   M = Modelo_Inicial
   DD não_rotulado = DD large
   for t=1 até T do
      DD pseudo = 0
      for all x in DD não_rotulado do
         y,confianca = Modelo(x)
         C_calibrado = CalibracaoModelo(c,M_calibrado)
         se MetadadoConfirma(y,x) então
            c = min(C_calibrado*fator_confianca,1)
         fim se 
         se fator_confianca >= limiar_confianca então
            DDpseudo = DDpseudo(x,y)
         fim se
      end for
      M = Ajuste Fino(M,DDpseudo)
      DD não_rotulado = DD não_rotulado \ DDpseudo
   end for
   return M
```

Para rodar o processo do algoritmo iterativo deve ser utilizado conforme o script abaixo:


```shell
python main_iterativo.py \
  --input Corpus_grande.json \
  --csv   comparacao_calibracao.csv \
  --iter  5 \
  --tau   0.80 \
  --alpha 1.2 \
  --model modelo_inicial \
  --out   saida_iterativa
```

Os parametros utilizados para rodar o processo foram :
- Input - Arquivo que deve ser lido
- CSV - Arquivo para ser feito de calibração
- Iter - Quantidade de iteração gerada
- tau - Valor do limiar de confiança
- alpha - Valor do acréscimo da confimação de metadados
- Model - Utilizado o modelo inicial
- out - A pasta com a saída com os resultados e os modelos

Para rodar esse processo há uma exigencia computacional portanto há uma pasta com o nome "separado" para evitar o gasto computacional. 

## Prerequisites 📋

Make sure you have the following prerequisites installed in your development environment:

- Python 3.x
- Pip (Python package manager)
- Torch
- Scikit-learn
- matplotlib
- Gliner

## How to Run 🏃‍♀️

Follow these simple steps to set up and run NERDD:

1. **Clone the Repository**

   ```shell
   git clone https://github.com/MLRG-CEFET-RJ/nerdd.git
   ```

1. **Install Libs on Environment**

   ```shell
   git install -r requirements.txt
   ```


## Contribution 🤝

If you want to contribute to the NERDD project, we would be happy to receive your contributions. Feel free to open issues or submit pull requests with improvements, bug fixes, or new features.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

We hope you enjoy offensive gesture detection with **GLINERDD**! If you have any questions or need assistance, please feel free to reach out to the development team. 😊👋