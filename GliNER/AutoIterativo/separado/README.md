# GLINERDD  -  AutoIterative 🚫

--------------------------------------------------------------------------------------------------------

**GLINERDD - AutoIterative - Separated** are the functional scripts present in the articles and using the present algorithm identified below:

```shell
M = Initial_Model
Unlabeled DD = DD large
for t=1 to T do
Pseudo DD = 0
for all x in Unlabeled DD do
y,confidence = Model(x)
C_calibrated = ModelCalibration(c, M_calibrated)
if MetadataConfirms(y,x) then
c = min(C_calibrated*confidence_factor,1)
end if
if confidence_factor >= confidence_threshold then
PseudoDD = PseudoDD(x,y)
end if
end for
M = FineTune(M, PseudoDD)
Unlabeled DD = Unlabeled DD \ PseudoDD
end for
return M
```

The scripts used for the following function:

- parte1_predication.py
This part performs the prediction of the defined model. To run the process, you must identify the following:
```shell
python part1_predication.py \
--input Corpus_grande.json \
--out Corpus_grande_001.json \
--model modelo_inicial
```
The parameters used to run the process were:

- Input - Input file for the prediction
- Out - Output file for the prediction
- Model - Model to be used
--------------------------------------------------------------------------------------------------------

- parte2_calibration.py

This part performs the calibration of the defined model, which can use the ts (Temperature Scaling) and iso (Isolation Regression) models. To run the process, you must identify the following:
```shell
python part2_calibration.py \
--preds_in Corpus_grande_001.json \
--preds_out Corpus_grande_001_calibradas.jsonl \
--calib_csv comparacao_calibracao.csv \
--method ts (or iso)
--preferred ts (or iso)
```
The parameters used to run the process were:

- Preds_in - Input file with the prediction
- Preds_out - Output file with the calibrated prediction
- Calib_csv - File to use for CSV calibration
- Method - Uses the TS or ISO method
- Preferred - Uses the used part defined as TS or ISO

--------------------------------------------------------------------------------------------------------

- parte3_avcalibracao.py

This part evaluates the calibration of the defined model, which can use the ts (Temperature Scaling) and iso (Isolation Regression) models. To run the process, you must identify Below:
```shell
python parte3_avcalibracao.py \
--calib_csv comparacao_calibracao.csv \
--out_dir ./avaliacao_calibracao \
--bins 10 --cv 5
```
The parameters used to run the process were:

- Calib_csv - File to use for CSV calibration
- Out_dir - Where the calibration metrics should be saved
- Bins - Separates into 10 different parts
- cv - Separates the cross validation

--------------------------------------------------------------------------------------------------------

- parte4_ajusteconfianca.py

This part confirms the metadata by expanding it using the selected factor of the defined model and comparing it with the metadata. To run the process, you must identify the following:
```shell
python parte4_confianca_v2.py \
--in Corpus_grande_001.jsonl \
--out Corpus_grande_001_confianca.jsonl \
--factor 1.2
--score_key score_relato
```
The parameters used to run the process were:

- in - File to be used as input in the confidence adjustment
- out - File to be used as output in the confidence adjustment
- factor - Value to be considered when multiplying by the confidence metadata factor
- score_key - Field that calculates the average confidence

--------------------------------------------------------------------------------------------------------

- parte5_separacao.py

This part separates the files based on confidence thresholds. To run the process, you must identify the following:
```shell
python3 parte5_separacao.py \
--in Corpus_grande_001_confianca.jsonl \
--out_dir ./separacao \
--score_field score_relato \
--thresh 0.8
```
The parameters used to The parameters used to run the process were:

- in - File to be used to input the confidence threshold
- out_dir - Folder to be used to separate the files
- score_field - Field to be considered for separation
- thresh - Value to be cut from the value

--------------------------------------------------------------------------------------------------------

- parte6_score_relato.py

This part separates the files based on the confidence thresholds. To run the process, you must identify the following:
```shell
python3 parte5_separacao.py \
--in Corpus_grande_001_confianca.jsonl \
--out_dir ./separacao \
--score_field score_relato \
--thresh 0.8
```
The parameters used to run the process were:

- in - File to be used to input the confidence threshold
- out_dir - Folder to be used

--------------------------------------------------------------------------------------------------------

- parte7_treino.py

This part fine-tunes the models to generate the model training. To run the process, you must identify the following:
```shell
python parte7_treino.py \
--input ./separacao_treino \
--out_dir ./gliner_finetuned_002 \
--base_model gliner_finetuned_001 \
--epochs 10 --patience 3 \
--device cuda --workers 2
```
The parameters used to run the process were:

- input - Folder containing the files to be used in training
- out_dir - Model to be saved in training
- base_model - Model to be used as a basis for training
- epochs - Number of epochs to be used
- patience - Threshold value to be used
- device - Select the cuda or CPU for which the training will be performed used
- workers - Selects the number of parts the process will be divided into

-----------------------------------------------------------------------------------------

- parte8_avaliacao.py

This part evaluates the models, generating the classification report and value predictions. To run the process, you must identify the following:
```shell
python parte8_avaliacao.py \
--input teste.json \
--model gliner_finetuned_002 \
--out_dir ./avaliacao_sanidade \
--labels "Person,Location,Organization"
```
The parameters used to run the process were:

- input - File that will be used to test the model
- model - Model to be used in the evaluation
- out_dir - Folder that will be used to save the metrics
- labels - Labels that will be used in this evaluation

--------------------------------------------------------------------------------------------------------

- parte8_replacelabel.py

This part supports the evaluation of the models to verify Which labels are used and replace the labels. To run the process, you must identify the following:
```shell
python parte8_replacelabel.py \
--in Corpus_grande_001_confianca.jsonl \
--out Corpus_grande_001_confianca_out.jsonl
--from_label Community
--to_label Location
--ci
--inplace

```
The parameters used to run the process were:

- in - Uses the file to modify the label
- out - Uses the file that can be changed in the label
- from_label - The label that would be searched
- to_label - The label to be changed
- ci - Marks case-sensitive
- inplace - If the file to be saved is the same

--------------------------------------------------------------------------------------------------------

- parte8_labelsdistintos.py

This part serves as support for evaluating the models to verify which labels were used. To run the process, you must identify Below:
```shell
python parte8_labelsdistintos.py \
--in Corpus_grande_001_confianca.jsonl
```
The parameters used to run the process were:

- in - Uses the file to be checked for labels

--------------------------------------------------------------------------------------------------------

- parte9_limpezaarquivos.py

This part serves to clean the files that were discarded, keeping only the fields necessary for a new iteration. To run the process, you must identify below:
```shell
python parte9_limpezaarquivos.py \
--in_dir iter001 \
--out_dir iter002

```
The parameters used to run the process were:

- in_dir - Uses the directory that needs to be cleaned
- out_dir - Uses the directory in the saved file

--------------------------------------------------------------------------------------------------------

**GLINERDD - AutoIterativo - Separado** são os scripts fuuncionais presentes nos artigos e utilizando o algoritmo presente que está identificado abaixo:  

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

Os scripts utilizados para a seguinte função:

- parte1_predicao.py 
Essa parte faz a predição do modelo definido.Para rodar o processo deve identificar abaixo:
```shell
   python parte1_predicao.py \
  --input Corpus_grande.json \
  --out  Corpus_grande_001.json \
  --model modelo_inicial
```
Os parametros utilizados para rodar o processo foram :

- Input - Arquivo de entrada que será feita a predição
- Out - Arquivo de saída que será a predição
- Model - Modelo que será utilizado
--------------------------------------------------------------------------------------------------------

- parte2_calibracao.py

Essa parte faz a calibração do modelo definido que pode ser utilizado o modelo ts (Temperature Scaling) e iso (Isolation Regression).Para rodar o processo deve identificar abaixo:
```shell
   python parte2_calibracao.py \
  --preds_in Corpus_grande_001.json \
  --preds_out Corpus_grande_001_calibradas.jsonl \
  --calib_csv comparacao_calibracao.csv \
  --method ts (ou iso) 
  --preferred ts (ou iso)
```
Os parametros utilizados para rodar o processo foram :

- Preds_in - Arquivo de entrada com a predição
- Preds_out - Arquivo de saída com a predição calibrada
- Calib_csv - Arquivo para ser utilizada calibração do CSV 
- Method - Utiliza o método TS ou ISO
- Preferred - Utiliza a parte utiliza definida como TS ou ISO

--------------------------------------------------------------------------------------------------------

- parte3_avcalibracao.py

Essa parte faz a avaliação da calibração do modelo definido que pode ser utilizado o modelo ts (Temperature Scaling) e iso (Isolation Regression).Para rodar o processo deve identificar abaixo:
```shell
   python parte3_avcalibracao.py \
  --calib_csv comparacao_calibracao.csv \
  --out_dir ./avaliacao_calibracao \
  --bins 10 --cv 5
```
Os parametros utilizados para rodar o processo foram :

- Calib_csv - Arquivo para ser utilizada calibração do CSV 
- Out_dir - Onde a pasta que devem ser salvas as métricas de calibração
- Bins - Separa em 10 partes diferentes
- cv - Separa o cross validation


--------------------------------------------------------------------------------------------------------

- parte4_ajusteconfianca.py

Essa parte faz a confirmação dos metadados ampliando através do fator selecionado do modelo definido e comparando com os metadados.Para rodar o processo deve identificar abaixo:
```shell
  python parte4_confianca_v2.py \
  --in Corpus_grande_001.jsonl \
  --out Corpus_grande_001_confianca.jsonl \
  --factor 1.2
  --score_key score_relato
```
Os parametros utilizados para rodar o processo foram :

- in - Arquivo para ser utilizada na entrada  no ajuste de confiança 
- out - Arquivo para ser utilizada na saída no ajuste de confiança 
- factor - Valor a ser considerado para multiplicar pelo fator de metadados de confiança
- score_key - Campo que cálcula a média das confianças

--------------------------------------------------------------------------------------------------------

- parte5_separacao.py

Essa parte faz a separação dos arquivos através dos limiares de confiança.Para rodar o processo deve identificar abaixo:
```shell
  python3 parte5_separacao.py \
  --in Corpus_grande_001_confianca.jsonl \
  --out_dir ./separacao \
  --score_field score_relato \
  --thresh 0.8
```
Os parametros utilizados para rodar o processo foram :

- in - Arquivo para ser utilizada na entrada no limiar de confiança 
- out_dir - Pasta a ser utilizada na separação dos arquivos 
- score_field - Campo que será considerado para ser feita a separação
- thresh - Valor que será cortado no valor

--------------------------------------------------------------------------------------------------------

- parte6_score_relato.py

Essa parte faz a separação dos arquivos através dos limiares de confiança.Para rodar o processo deve identificar abaixo:
```shell
  python3 parte5_separacao.py \
  --in Corpus_grande_001_confianca.jsonl \
  --out_dir ./separacao \
  --score_field score_relato \
  --thresh 0.8
```
Os parametros utilizados para rodar o processo foram :

- in - Arquivo para ser utilizada na entrada no limiar de confiança 
- out_dir - Pasta a ser utilizada na separação dos arquivos 
- score_field - Campo que será considerado para ser feita a separação
- thresh - Valor que será cortado no valor

--------------------------------------------------------------------------------------------------------

- parte7_treino.py 

Essa parte faz os ajustes fino dos modelos para gerar o treino do modelo .Para rodar o processo deve identificar abaixo:
```shell
  python parte7_treino.py \
  --input ./separacao_treino \
  --out_dir ./gliner_finetuned_002 \
  --base_model gliner_finetuned_001  \
  --epochs 10 --patience 3 \
  --device cuda --workers 2
```
Os parametros utilizados para rodar o processo foram :

- input - Pasta que possui os arquivos a serem utilizados no treino 
- out_dir - Modelo a ser salvo no treinamento 
- base_model - Modelo a ser utilizado como base para o treinamento
- epochs - Quantidade de épocas a serem utilizadas
- patience - Valor de limite que será utilizado 
- device - Seleciona o cuda ou CPU para qual sera utilizada
- workers - Seleciona em quantas as partes será dividido o processo

--------------------------------------------------------------------------------------------------------

- parte8_avaliacao.py

Essa parte faz a avaliação dos modelos gerando o classification_report e as predições dos valores.Para rodar o processo deve identificar abaixo:
```shell
  python parte8_avaliacao.py \
  --input teste.json \
  --model gliner_finetuned_002 \
  --out_dir ./avaliacao_sanidade \
  --labels "Person,Location,Organization"
```
Os parametros utilizados para rodar o processo foram :

- input - Arquivo que será utilizado no teste do modelo
- model - Modelo a ser utilizado na avaliação 
- out_dir - Pasta que será utilizada para gravar as métricas
- labels -  Labels que serão utilizados nessa avaliação

--------------------------------------------------------------------------------------------------------

- parte8_replacelabel.py

Essa parte serve de apoio para a avaliação dos modelos para verificar quais os labels utilizados e substituir os labels.Para rodar o processo deve identificar abaixo:
```shell
  python parte8_replacelabel.py \
  --in Corpus_grande_001_confianca.jsonl \
  --out Corpus_grande_001_confianca_out.jsonl
  --from_label  Comunidade
  --to_label    Location
  --ci 
  --inplace 

```
Os parametros utilizados para rodar o processo foram :

- in - Utiliza o arquivo a ser modificado o label
- out - Utiliza o arquivo que pode ser alterado no label
- from_label - O label que seria procurado
- to_label - O label a ser alterado
- ci - Faz a marcação do case sensitive
- inplace - Caso o arquivo a ser salvo seja o mesmo

--------------------------------------------------------------------------------------------------------

- parte8_labelsdistintos.py

Essa parte serve de apoio para a avaliação dos modelos para verificar quais os labels utilizados.Para rodar o processo deve identificar abaixo:
```shell
    python parte8_labelsdistintos.py \
  --in Corpus_grande_001_confianca.jsonl
```
Os parametros utilizados para rodar o processo foram :

- in - Utiliza o arquivo a ser verificado para os labels

--------------------------------------------------------------------------------------------------------

- parte9_limpezaarquivos.py

Essa parte serve para fazer a limpeza nos arquivos que foram descartados mantendo somente os campos necessários para fazer uma nova iteração.Para rodar o processo deve identificar abaixo:
```shell
  python parte9_limpezaarquivos.py \
  --in_dir iter001 \
  --out_dir iter002

```
Os parametros utilizados para rodar o processo foram :

- in_dir - Utiliza o diretório que precisa ser limpo
- out_dir - Utiliza o diretório no arquivo que foi salvo

--------------------------------------------------------------------------------------------------------


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