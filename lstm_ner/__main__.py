import os
import sys

from functools import reduce
from typing import Dict, List, Tuple

from keras.models import load_model
from tensorflow.python.keras import utils
#from keras.utils import np_utils 
#from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import numpy as np

from utils import data_utils
import ner_model as ner

# defining constants
word_embeddings_file = 'data/cbow_s50.txt'
#word_embeddings_file = 'data/vectors_W3_D50.txt'
input_data_folder = 'dataset2'
model_file = 'output/model2.h5'
char_embeddings_file = 'output/char_embeddings.txt'

# defining hyper parameters
word_window_size = 1
char_window_size = 1
char_embeddings_dim = 20
dropout_rate = 0.5
lstm_units = 420
conv_num = 10
epochs = 2
test_percent = 0.9
#not_entity_threshold = 0.7


def main():
    # getting args
    cpu_only = '--cpu-only' in sys.argv

    # loading data from files
    word_embeddings, word2idx, char2idx = data_utils.read_embeddings_file(word_embeddings_file)
    max_word_len = max(map(lambda word: len(word), word2idx.keys()))
    train_data, test_data, label2idx = data_utils.load_dataset(input_data_folder, test_percent)
    print('train sentences:', len(train_data))
    print('test sentences:', len(test_data))
    print("epochs: ", epochs)
    # train_data = train_data[:50]
    # test_data = test_data[:10]
    x_train, y_train = data_utils.transform_to_xy(train_data, word2idx, label2idx, word_window_size,
                                                  char2idx, max_word_len)
    x_test, y_test = data_utils.transform_to_xy(test_data, word2idx, label2idx, word_window_size,
                                                char2idx, max_word_len)
    num_labels = len(label2idx)
    # "binarize" labels
    y_train = to_categorical(y_train, num_labels)
    y_test = to_categorical(y_test, num_labels)
    # load model whether it is saved
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

        # summarize the model
        print(model.summary())

        # training model
        model.fit(x_train, y_train, epochs=epochs)

        # saving embeddings
        # embedding_layer = char_embedding_model.layers[0]
        # weights = embedding_layer.get_weights()[0]
        # data_utils.save_embeddings(char_embeddings_file, weights, char2idx)

        # saving whole model
        # model.save(model_file)

    # evaluating model
    print('x_test:')
    print(x_test)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('y_test:')
    print(y_test)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    _, accuracy, precision, recall = model.evaluate(x_test, y_test)
    print('Accuracy: %f' % (accuracy * 100))
    print('Precision: %f' % (precision * 100))
    print('Recall: %f' % (recall * 100))


    # make predictions
    output = model.predict(x_test)
    trainPredict = model.predict(x_train)
    testPredict = model.predict(x_test)
    testPredict = np.argmax(output, axis=1)
    y_test = np.argmax(y_test, axis=1)

    predicted = model

    train_data_flat = reduce(lambda acc, cur: acc + cur, train_data, [])
    label_dist = {label: 0 for label in label2idx.keys()}
    for _, label in train_data_flat:
        label_dist[label] += 1
    print()
    print('####### train label distribution')
    print('total: %d\n' % len(train_data_flat))
    for label, count in label_dist.items():
        print(label, count)
    print()

    test_data_flat = reduce(lambda acc, cur: acc + cur, test_data, [])
    print('####### test label distribution')
    print('total: %d\n' % len(test_data_flat))
    label_dist = {label: 0 for label in label2idx.keys()}
    for _, label in test_data_flat:
        label_dist[label] += 1
    for label, count in label_dist.items():
        print(label, count)
    print()
    cm = confusion_matrix(y_test, testPredict)
    print(cm)

def label_output(output: List[float], label2idx: Dict[str, int], test_data_flat: List[Tuple[str, str]]):
    classed_output = []
    for i in range(len(output)):
        not_entity_idx = label2idx['O']
        # if output[i, not_entity_idx] >= not_entity_threshold:
        #     entity = 'O'
        # else:
        ent_prob_max = 0
        ent_idx = not_entity_idx
        for j, ent in enumerate(output[i]):
            #     if ent > ent_prob_max and j != not_entity_idx:
            if ent > ent_prob_max:
                ent_prob_max = ent
                ent_idx = j
        entity = [label for label, idx in label2idx.items() if idx == ent_idx][0]
        classed_output.append((test_data_flat[i][0], entity))
    return classed_output


if __name__ == '__main__':
    main()