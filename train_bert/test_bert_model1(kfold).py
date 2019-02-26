# -*- coding: utf-8 -*-

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import os
import io
import glob
import numpy as np
import pandas as pd
import csv
import itertools
import json
import pickle

import logging
import logging.handlers

import sklearn.metrics
from sklearn.model_selection import StratifiedKFold

import keras
from keras.layers import Lambda
from keras.layers.merge import add, multiply
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input
from keras.layers import recurrent
from keras.layers import Flatten
from keras.layers.core import RepeatVector, Dense
from keras.layers.wrappers import Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.models import model_from_json

from keras_bert import get_base_dict, get_model, gen_batch_inputs
import keras_contrib.optimizers
from keras_contrib.optimizers import FTML

import hyperopt
from hyperopt import hp, tpe, STATUS_OK, Trials

import sentencepiece as spm


# алгоритм сэмплирования гиперпараметров
HYPEROPT_ALGO = tpe.suggest  # tpe.suggest OR hyperopt.rand.suggest


PAD_CHAR = u''

tmp_folder = '../tmp'
best_params_path = os.path.join(tmp_folder, 'test_bert_model1(kfold).best_params.json')


#REPRES = 'chars'
REPRES = 'sentencepiece'


sp = spm.SentencePieceProcessor()
rc = sp.Load(os.path.join(tmp_folder, "sentencepiece4bert_8000.model"))


def init_trainer_logging(logfile_path):
    # настраиваем логирование в файл и эхо-печать в консоль
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    lf = logging.FileHandler(logfile_path, mode='w')

    lf.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    lf.setFormatter(formatter)
    logging.getLogger('').addHandler(lf)


def sent2pieces(sent):
    return sp.EncodeAsPieces(sent)


def create_model(params, bert_config):
    input = Input(shape=(bert_config['seq_len'], bert_config['embed_dim'],), dtype='float32', name='input')
    net = input

    arch = params['net_arch']
    if arch == 'ff':
        net = Flatten()(net)
        net = Dense(units=params['units1'], activation=params['activation1'])(net)
    elif arch == 'lstm':
        net = Bidirectional(recurrent.LSTM(units=params['units1'],
                                           dropout=params['dropout_rate'],
                                           return_sequences=False))(net)
    elif arch == 'cnn(1)':
        merging_layers = []
        nb_filters = params['nb_filters']
        for kernel_size in range(params['min_kernel_size'], params['max_kernel_size']+1):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1,
                          name='conv_{}'.format(kernel_size))

            if params['pooling'] == 'average':
                pooler = keras.layers.GlobalAveragePooling1D()
            elif params['pooling'] == 'max':
                pooler = keras.layers.GlobalMaxPooling1D()
            else:
                raise NotImplementedError()

            conv_layer1 = conv(net)

            if params['dropout_rate'] > 0.0:
                conv_layer1 = keras.layers.Dropout(rate=params['dropout_rate'])(conv_layer1)

            conv_layer1 = pooler(conv_layer1)
            merging_layers.append(conv_layer1)

        net = keras.layers.concatenate(inputs=merging_layers)
    elif arch == 'cnn(2)':
        merging_layers = []
        nb_filters = 64
        for kernel_size in [2, 3, 4, 5]:
            stream = Conv1D(filters=nb_filters,
                            kernel_size=3,
                            padding='valid',
                            activation='relu',
                            strides=1,
                            name='conv1_{}'.format(kernel_size))(net)

            stream = keras.layers.AveragePooling1D(pool_size=3, strides=None, padding='valid',
                                                   data_format='channels_last')(stream)

            stream = Conv1D(filters=64,
                            kernel_size=3,
                            padding='valid',
                            activation='relu',
                            strides=1,
                            name='conv2_{}'.format(kernel_size))(stream)

            stream = keras.layers.GlobalMaxPooling1D()(stream)
            merging_layers.append(stream)

        net = keras.layers.concatenate(inputs=merging_layers)
    elif arch == 'lstm(cnn)':
        merging_layers = []
        encoder_size = 0
        nb_filters = 64
        rnn_size = 100
        for kernel_size in range(1, 4):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1,
                          name='shared_conv_{}'.format(kernel_size))

            # pooler = keras.layers.GlobalAveragePooling1D()
            lstm = recurrent.LSTM(rnn_size, return_sequences=False)
            pooler = keras.layers.AveragePooling1D(pool_size=kernel_size, strides=None, padding='valid')

            conv_layer1 = conv(net)
            conv_layer1 = pooler(conv_layer1)
            conv_layer1 = lstm(conv_layer1)
            merging_layers.append(conv_layer1)
            encoder_size += nb_filters

        net = keras.layers.concatenate(inputs=merging_layers)



    if params['normalization'] == 'BatchNormalization':
        net = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,
                                              center=True, scale=True)(net)
    elif params['normalization'] == 'GroupNormalization':
        net = keras_contrib.layers.GroupNormalization(groups=32)(net)
    elif params['normalization'] == '':
        pass
    else:
        raise NotImplementedError()

    net = Dense(units=params['units2'], activation=params['activation2'])(net)

    net = Dense(units=1, activation='sigmoid')(net)

    model = Model(inputs=[input], outputs=net)

    if params['optimizer'] == 'ftml':
        opt = keras_contrib.optimizers.FTML()
    else:
        opt = params['optimizer']

    model.compile(loss='binary_crossentropy', optimizer=opt)

    #model.summary()

    return model


def train_model(params, model, X_train, y_train, X_val, y_val):
    weights_path = '../tmp/model1.weights'
    monitor_metric = 'val_loss'
    model_checkpoint = ModelCheckpoint(weights_path, monitor=monitor_metric, verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')

    callbacks = [model_checkpoint, early_stopping]

    model.fit(x=X_train, y=y_train,
              validation_data=(X_val, y_val),
              epochs=1000,
              batch_size=params['batch_size'],
              verbose=2,
              callbacks=callbacks)

    model.load_weights(weights_path)
    y_pred = model.predict(X_val)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_val, y_pred)
    threshold = thresholds[np.argmax(tpr - fpr)]

    y_pred2 = (y_pred >= threshold).astype(np.int)
    acc = sklearn.metrics.accuracy_score(y_true=y_val, y_pred=y_pred2)
    print('threshold={} accuracy={}'.format(threshold, acc))

    return acc


def estimate(params, bert_config):
    kfold_acc_sum = 0.0
    kfold_acc_denom = 0.0
    kfold_accs = []

    logging.info('Start estimating params={}'.format(get_params_str(params)))

    kf = StratifiedKFold(n_splits=3)
    for ifold, (train_index, test_index) in enumerate(kf.split(X_data, y_data)):

        X_train, X_val = X_data[train_index], X_data[test_index]
        y_train, y_val = y_data[train_index], y_data[test_index]

        for probe in range(1):  # поставить >1, если будем усреднять несколько запусков нейросетки
            model = create_model(params, bert_config)
            acc = train_model(params, model, X_train, y_train, X_val, y_val)
            logging.info('KFOLD[{}] --> acc={:6.4f}'.format(ifold, acc))
            kfold_acc_sum += acc
            kfold_acc_denom += 1.0
            kfold_accs.append(acc)

    acc = kfold_acc_sum / kfold_acc_denom
    std = np.std(kfold_accs)

    return acc, std


def get_params_str(params):
    return u' '.join('{}={}'.format(p, v) for (p, v) in params.items())


cur_best_acc = 0.0
def ho_objective(space):
    global cur_best_acc

    params = dict(filter(lambda kv: kv[0] != 'net_arch_type', space.items()))
    params.update(space['net_arch_type'].items())

    acc, std = estimate(params, bert_config)
    logging.info('Accuracy mean={:6.4f} std={:6.4f}'.format(acc, std))

    if acc > cur_best_acc:
        cur_best_acc = acc
        best_params = params
        logging.info('!!! New best score={} for params={}'.format(acc, get_params_str(params)))
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f)

        wrt_best.write(
            u'\n\n{}\nAccuracy mean={:6.4f} std={:6.4f}\n'.format(get_params_str(best_params), acc, std))
        wrt_best.flush()

    return acc


init_trainer_logging(os.path.join(tmp_folder, 'test_bert_model1(kfold).log'))
logging.info('Start')


with open('../tmp/bert.config', 'r') as f:
    bert_config = json.load(f)

max_seq_len = bert_config['seq_len']

weights_path = '../tmp/bert.weights'

inputs, output_layer = get_model(training=False, **bert_config)
bert_model = Model(inputs=inputs, outputs=output_layer)
bert_model.load_weights(weights_path)

with open('../tmp/token_dict.pickle', 'r') as f:
    token_dict = pickle.load(f)



samples0 = []
max_text_len = 0
for file in glob.glob(os.path.join('../data', '*.csv')):
    print('Loading samples from {}...'.format(file))
    df = pd.read_csv(file, encoding='utf-8', delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for irow, row in df.iterrows():
        text = row['text']
        label = row['class']
        pieces = sent2pieces(text)[:max_seq_len]
        samples0.append((text, label))
        max_text_len = max(max_text_len, len(pieces))

nb_samples = len(samples0)

print('{} samples loaded'.format(nb_samples))

token_input = np.zeros((nb_samples, max_seq_len), dtype=np.int32)
seg_input = np.zeros((nb_samples, max_seq_len), dtype=np.int32)

for isample, (text, label) in enumerate(samples0):
    tokens = sent2pieces(text)[:max_seq_len]
    #token_input1 = [token_dict.get(token, 0) for token in tokens] + [0] * (max_seq_len - len(tokens))
    #seg_input1 = [0] * len(tokens) + [0] * (max_seq_len - len(tokens))
    #token_input.append(token_input1)
    #seg_input.append(seg_input1)
    for itoken, token in enumerate(tokens):
        if token not in token_dict:
            print(u'Token {} missing in token_dict'.format(token))
        else:
            token_input[isample, itoken] = token_dict[token]

#token_input = np.asarray(token_input)
#seg_input = np.asarray(seg_input)

print('Vectorization of {} samples'.format(nb_samples))

y_pred = bert_model.predict([token_input, seg_input])
X_data = y_pred
y_data = np.zeros(nb_samples, dtype=np.bool)

for isample, (text, label) in enumerate(samples0):
    y_data[isample] = label

nb_0 = sum(y_data == 0)
nb_1 = sum(y_data == 1)
print('nb_0={}'.format(nb_0))
print('nb_1={}'.format(nb_1))

wrt_best = io.open(os.path.join(tmp_folder, 'test_bert_model1(kfold).best.txt'), 'w')

#params = dict()
#acc, std = estimate(params, bert_config)
#print('acc={} std={}'.format(acc, std))

space = hp.choice('repres_type', [
    {
        'optimizer': hp.choice('optimizer', ['adam', 'nadam', 'ftml']),
        'batch_size': hp.choice('batch_size', range(100, 400, 50)),
        'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.6),
        'units2': hp.choice('units2', range(5, 30)),
        'activation2': hp.choice('activation2', ['sigmoid', 'relu']),  # 'pelu'
        'normalization': hp.choice('normalization',
                                   ['', 'BatchNormalization']),  # 'GroupNormalization', 'InstanceNormalization'

        'net_arch_type': hp.choice('net_arch_type', [
            {'net_arch': 'lstm',
             'units1': hp.choice('units1', range(64, 256))},
            {'net_arch': 'cnn(1)',
             'pooling': hp.choice('pooling_cnn1', ['max']),  # возможен варант 'average'
             'nb_filters': hp.choice('nb_filters_cnn1', [32, 64, 96, 128]),
             'min_kernel_size': hp.choice('min_kernel_size_cnn1', [1]),
             'max_kernel_size': hp.choice('max_kernel_size_cnn1', [2, 3, 4]),
             },
            # todo
        ]),
    },
])

trials = Trials()
best = hyperopt.fmin(fn=ho_objective,
                     space=space,
                     algo=HYPEROPT_ALGO,
                     max_evals=100,
                     trials=trials,
                     verbose=1)




