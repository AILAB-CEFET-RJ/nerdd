# -*- coding: utf-8 -*-
# Import das Bibliotecas
import os
import sys
from functools import reduce
from typing import Dict, List, Tuple
from keras.models import load_model
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collections import Counter

import csv  
import sys
import re
import pandas as pd
import glob
from builtins import Exception
from typing import List, Dict, Tuple
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from utils import text_utils
import string
from functools import reduce
import tensorflow as tf
import nltk
import re
#from unidecode import unidecode
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
nltk.download('punkt')
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras import Sequential, Model
from keras.layers import Embedding, LSTM, Dropout, Dense, Reshape, Conv1D, MaxPooling1D, TimeDistributed, \
    concatenate
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import EarlyStopping

from utils import data_utils
import ner_model as ner

# defining constants
# word_embeddings_file = 'data/cbow_s50.txt'
word_embeddings_file = 'data/vectors_W3_D50.txt'
input_data_folder = 'data'
# model_file = 'output/model-cbow.h5'
model_file = 'output/model2.h5'
char_embeddings_file = 'output/char_embeddings.txt'
# defining hyper parameters
word_window_size = 2
char_window_size = 5
char_embeddings_dim = 20
dropout_rate = 0.5
lstm_units = 420
conv_num = 10
epochs = 200
test_percent = 0.2
#not_entity_threshold = 0.7

# Classe com as métricas acurácia, precisão e f1-score
""" class Metrics:
    def __init__(self):
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
        self.actual_total = 0

    def total_predicted(self):
        return self.true_pos + self.true_neg + self.false_pos + self.false_neg

    def accuracy(self):
        return (self.true_pos + self.true_neg) / self.actual_total

    def precision(self):
        if self.true_pos + self.false_pos == 0:
            return 0
        return self.true_pos / (self.true_pos + self.false_pos)

    def recall(self):
        if self.true_pos + self.false_neg == 0:
            return 0
        return self.true_pos / (self.true_pos + self.false_neg)

    def f_measure(self):
        precision, recall = self.precision(), self.recall()
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)    

 """

def f1_score_metric(y_true, y_pred):
    return f1_score(y_true, np.round(y_pred), average='macro',pos_label=1)


cpu_only = '--cpu-only' in sys.argv

# loading data from files
word_embeddings, word2idx, char2idx = data_utils.read_embeddings_file(word_embeddings_file)

max_word_len = max(map(lambda word: len(word), word2idx.keys()))

# Separa as bases
train_data, test_data, label2idx = data_utils.load_dataset_sklearn(input_data_folder, test_percent)

print('train sentences:', len(train_data))
print('test sentences:', len(test_data))
print("epochs: ", epochs)

# transforma em X e Y e coloca a janela de contexto
x_test, y_test = data_utils.transform_to_xy(test_data, word2idx, label2idx, word_window_size,char2idx, max_word_len)
# transforma em X e Y e coloca a janela de contexto
x_train, y_train = data_utils.transform_to_xy(train_data, word2idx, label2idx, word_window_size,char2idx, max_word_len)
# labels pelas entidades
num_labels = len(label2idx)
# "binarize" labels
y_train = to_categorical(y_train, num_labels)
y_test = to_categorical(y_test, num_labels)

if os.path.exists(model_file):
    model = load_model(model_file)
    print(f'Model loaded from {model_file}')
    print(model.summary())
else:
    # defining model
    word_input_length = 2 * word_window_size + 1
    max_word_len_padded = max_word_len + word_window_size * 2
    word_embedding_model = ner.generate_word_embedding_model(word_input_length, weights=word_embeddings)
    char_embedding_model = ner.generate_char_embedding_model(max_word_len, max_word_len_padded, word_input_length,
                                                                 char_embeddings_dim, conv_num, char_window_size,
                                                                 vocab_size=len(char2idx))
    model = ner.generate_model(word_embedding_model, char_embedding_model, lstm_units, num_labels, dropout_rate,
                                    cpu_only=cpu_only)


class F1ScoreEarlyStopping(Callback):
    def __init__(self, validation_data=(), margin=0.02, patience=20):
        super(Callback, self).__init__()
        self.validation_data = validation_data
        self.margin = margin
        self.patience = patience
        self.best_f1_score = -1
        self.wait = 0
        
    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val)
        f1_val = f1_score(y_val, np.round(y_pred), average='macro')
        print(f" - val_f1_score: {f1_val:.4f}")
        calculo_f1 = self.best_f1_score + self.margin
        if self.best_f1_score == -1 or f1_val > self.best_f1_score:
            self.best_f1_score = f1_val
            self.wait = 0
            print(f" - best f1 score: {self.best_f1_score:.4f}")
            print(f" - wait: {self.wait:.4f}")
            #print(f" - calculo: {calculo_f1:.4f}")
        else:
            self.wait += 1
            print(f" - best f1 score: {self.best_f1_score:.4f}")
            print(f" - wait: {self.wait:.4f}")
            #print(f" - calculo: {calculo_f1:.4f}")
            if self.wait >= self.patience:
                print("Early stopping due to no improvement in F1-Score.")
                self.model.stop_training = True
            else:
                print(f"No improvement in F1-Score for {self.wait} epochs.")

callback = F1ScoreEarlyStopping(validation_data=(x_test, y_test), margin=0.02, patience=20)

# Treinar o modelo
history = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_test, y_test), verbose=1,callbacks=[callback])
# history = model.fit(x_train, y_train, epochs=300, batch_size=32, validation_data=(x_test, y_test), verbose=1)

print(history.history)

# Avaliar o modelo
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plotar a curva de perda durante o treinamento e validação
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Grafico- Com EarlyStopping .png") 
#plt.show()
output = model.predict(x_test)
print(output[:5])  # Print first 5 predictions to check their values
print(y_test[:5])
output_labels = np.argmax(output, axis=1)  # Pega a classe de maior valor/probabilidade
y_test_labels = np.argmax(y_test, axis=1)  # Se y_test for codificado como one-hot

cr = classification_report(output_labels, y_test_labels)
print(cr)



# saving whole model
model.save(model_file)