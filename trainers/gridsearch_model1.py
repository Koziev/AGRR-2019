# -*- coding: utf-8 -*-

"""
Полуавтоматический подбор параметров нейросетевой модели, классифицирующей предложения
по признаку наличия пропуска (столбец class в датасете) для задачи
AGRR-2019 (https://github.com/dialogue-evaluation/AGRR-2019)

Проверяются следующие варианты представления текста:
1) слова
2) символы
3) элементы sentencepiece
4) слоги

Варианты архитектуры нейросетки:
1) сверточная (1 и 2 слоя сверток)
2) рекуррентная

Подбор параметров выполняется либо исчерпывающим поиском по сетке,
либо с помощью hyperopt.

(c) Koziev Ilya inkoziev@gmail.com
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import os
import io
import gc
import numpy as np
import pandas as pd
import operator
import csv
import itertools
import re
import json
import platform

import logging
import logging.handlers

import gensim
from gensim.models.wrappers import FastText

import sklearn.metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json

import keras_contrib.layers.normalization
#from keras_contrib.layers.advanced_activations import PELU
import keras_contrib.optimizers
from keras_contrib.optimizers import FTML

import hyperopt
from hyperopt import hp, tpe, STATUS_OK, Trials

import sentencepiece as spm
import ruword2tags
import rusyllab
import rupostagger

from agrr_tokenizer import TokenizerAGRR


MAX_TEXT_LEN = 200  # макс. число символов в тексте; если больше - не используем для обучения

MAX_NB_SAMPLES = 25000  # макс. число сэмплов, используемых для обучения

# Выполнить поиск оптимальных параметров с помощью hyperopt
do_hyperopt = False

# Выполнить поиск оптимальных параметров по сетке (рекомендуемый способ - hyperopt).
do_gridsearch = False

# Взять лучшие параметры (из сценария do_gridsearch или do_hyperopt),
# создать модель, обучить ее, обработать входной файл, сохранить
# результаты в csv.
run_best = True

PAD_CHAR = u''
PAD_FEATURE = u'<padder>'

#data_files = ['train.csv', 'dev.csv', 'add.csv']
#data_files = ['train.csv', 'add.csv']
data_files = ['train.csv', 'dev.csv']
#data_files = ['train.csv']
tmp_folder = '../tmp'
best_params_path = os.path.join(tmp_folder, 'gridsearch_model1.best_params.json')

# алгоритм сэмплирования гиперпараметров
HYPEROPT_ALGO = tpe.suggest  # tpe.suggest OR hyperopt.rand.suggest


def is_linux():
    return platform.system() == 'Linux'


def get_params_str(params):
    return u' '.join('{}={}'.format(p, v) for (p, v) in params.items())


def is_num(token):
    return re.match('^[0-9]+$', token)


def get_word_features(word, postags, word2tags, w2v, wc2v):
    features = set()

    if is_num(word):
        features.add(('<number>', 1.0))
    elif word[0] in u'‼≠™®•·[¡+<>`~;.,‚?!-…№”“„{}|‹›/\'"–—_:«»*]()‘’≈':
        features.add((u'punct_{}'.format(ord(word[0])), 1.0))
    elif word[0] in u'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            features.add((u'<latin>', 1.0))
    else:
        for tagset in word2tags[word]:
            for tag in tagset.split(' '):
                features.add((tag, 1.0))

    lword = word.lower()
    if lword in w2v:
        v = w2v[lword]
        for ix, x in enumerate(v):
            features.add((u'w2v[{}]'.format(ix), x))
    else:
        features.add((u'<oov_w2v>', 1.0))

    if True:
        if word in wc2v:
            v = wc2v[word]
            for ix, x in enumerate(v):
                features.add((u'wc2v[{}]'.format(ix), x))

    if postags:
        for tag in postags.split('|'):
            features.add((tag, 1.0))

    return list(features)


def sent2pieces(spm_splitter, tokenizer, sent, max_sent_len, repres):
    if repres == 'sentencepiece':
        px = spm_splitter.EncodeAsPieces(sent)
    elif repres == 'chars':
        px = list(sent)
    elif repres == 'syllables':
        words = tokenizer.tokenize(sent)
        px = rusyllab.split_words(words)
    else:
        raise NotImplementedError()

    if max_sent_len == 0:
        return px
    else:
        l = len(px)
        if l < max_sent_len:
            return [u'\b'] + px + [u'\n'] + list(itertools.repeat(PAD_CHAR, (max_sent_len - l)))
        else:
            return [u'\b'] + px + [u'\n']


def init_trainer_logging(logfile_path):
    # настраиваем логирование в файл и эхо-печать в консоль
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    lf = logging.FileHandler(logfile_path, mode='w')

    lf.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    lf.setFormatter(formatter)
    logging.getLogger('').addHandler(lf)


def create_model(params, computed_params):
    max_text_len = computed_params['max_text_len']

    if params['repres'] == 'words':
        nb_features = computed_params['nb_features']
        input = Input(shape=(max_text_len, nb_features,), dtype='float32', name='input')
        net = input
    else:
        nb_pieces = computed_params['nb_pieces']
        piece_dim = params['piece_dim']

        input = Input(shape=(max_text_len,), dtype='int32', name='input')

        if params['piece_dim'] == -1:
            # Инициализируем матрицу эмбеддингов единичной диагональную.
            # Эти эмбединги не будут меняться.
            embeddings = np.zeros((nb_pieces, nb_pieces), dtype=np.float32)
            for i in range(nb_pieces):
                embeddings[i, i] = 1.0
            net = keras.layers.Embedding(nb_pieces, nb_pieces,
                                         input_length=max_text_len,
                                         trainable=False,
                                         weights=[embeddings])(input)
        else:
            net = keras.layers.Embedding(nb_pieces,
                                         piece_dim,
                                         input_length=max_text_len,
                                         trainable=True)(input)

    if params['net_arch'] == 'cnn(1)':
        # Один сверточный слой
        merging_layers = []
        for kernel_size in range(params['min_kernel_size'], params['max_kernel_size'] + 1):
            conv = Conv1D(filters=params['nb_filters'],
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

        if len(merging_layers) > 1:
            net = keras.layers.concatenate(inputs=merging_layers)
        else:
            net = merging_layers[0]
    elif params['net_arch'] == 'cnn(2)':
        # Два сверточных слоя
        merging_layers = []
        for kernel_size in range(params['min_kernel_size'], params['max_kernel_size'] + 1):
            conv = Conv1D(filters=params['nb_filters'],
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1,
                          name='conv1_{}'.format(kernel_size))

            if params['pooling'] == 'average':
                pooler = keras.layers.AveragePooling1D(pool_size=3,
                                                       strides=None,
                                                       padding='valid',
                                                       data_format='channels_last')
            elif params['pooling'] == 'max':
                pooler = keras.layers.MaxPooling1D(pool_size=3,
                                                   strides=None,
                                                   padding='valid',
                                                   data_format='channels_last')
            else:
                raise NotImplementedError()

            stream = conv(net)

            if params['dropout_rate'] > 0.0:
                stream = keras.layers.Dropout(rate=params['dropout_rate'])(stream)

            stream = pooler(stream)

            stream = Conv1D(filters=params['nb_filters'],
                            kernel_size=2,
                            padding='valid',
                            activation='relu',
                            strides=1,
                            name='conv2_{}'.format(kernel_size))(stream)

            if params['pooling'] == 'average':
                stream = keras.layers.GlobalAveragePooling1D()(stream)
            elif params['pooling'] == 'max':
                stream = keras.layers.GlobalMaxPooling1D()(stream)
            else:
                raise NotImplementedError()

            merging_layers.append(stream)

        if len(merging_layers) > 1:
            net = keras.layers.concatenate(inputs=merging_layers)
        else:
            net = merging_layers[0]
    elif params['net_arch'] == 'lstm':
        net = Bidirectional(recurrent.LSTM(units=params['units1'],
                                           dropout=params['dropout_rate'],
                                           return_sequences=False))(net)
    else:
        raise NotImplementedError('net_arch="{}" is not implemented'.format(net_arch))

    if params['activation2'] == 'pelu':
        net = Dense(units=params['units2'], activation=keras_contrib.layers.advanced_activations.PELU)(net)
    else:
        net = Dense(units=params['units2'], activation=params['activation2'])(net)

    if params['normalization'] == 'BatchNormalization':
        net = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,
                                              center=True, scale=True)(net)
    elif params['normalization'] == 'GroupNormalization':
        net = keras_contrib.layers.GroupNormalization(groups=32)(net)
    elif params['normalization'] == '':
        pass
    else:
        raise NotImplementedError()


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
    monitor_metric = 'val_loss'
    weights_path = os.path.join(tmp_folder, 'model1.weights')
    arch_path = os.path.join(tmp_folder, 'model1.arch')

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


w2v_storage = dict()  # кэш загруженных моделей w2v
word_samples0 = None

def prepare_data(params, pred_filepath, wc2v):
    global word_samples0

    computed_params = dict()
    pred_samples = None
    X_pred = None
    splitter = None

    if params['repres'] == 'sentencepiece':
        sentencepiece_corpus = os.path.join(tmp_folder, 'sentencepiece_corpus.txt')
        spm_name = 'enru_model{}'.format(params['spm_items'])

        if not os.path.exists(os.path.join(tmp_folder, spm_name + '.vocab')):
            spm.SentencePieceTrainer.Train(
                '--input={} --model_prefix={} --vocab_size={} --character_coverage=1.0 --model_type=bpe'.format(
                    sentencepiece_corpus, spm_name, params['spm_items']))
            os.rename(spm_name + '.vocab', os.path.join(tmp_folder, spm_name + '.vocab'))
            os.rename(spm_name + '.model', os.path.join(tmp_folder, spm_name + '.model'))

        splitter = spm.SentencePieceProcessor()
        splitter.Load(os.path.join(tmp_folder, spm_name + '.model'))

    if params['repres'] in ['sentencepiece', 'chars', 'syllables']:
        samples0 = []
        max_text_len = 0
        #for file in glob.glob(os.path.join('../data', '*.csv')):
        for file0 in data_files:
            file = os.path.join('../data', file0)
            if os.path.exists(file):
                logging.info(u'Loading samples from {}...'.format(file))
                df = pd.read_csv(file, encoding='utf-8', delimiter='\t', quoting=csv.QUOTE_MINIMAL)
                logging.info('df.shape={}'.format(df.shape))
                for irow, row in df.iterrows():
                    text = row['text']
                    if len(text) <= MAX_TEXT_LEN:
                        label = row['class']
                        pieces = sent2pieces(splitter, tokenizer, text, 0, params['repres'])
                        samples0.append((text, label))
                        max_text_len = max(max_text_len, len(pieces))

                        if len(samples0) >= MAX_NB_SAMPLES:
                            break

        logging.info('{} samples loaded'.format(len(samples0)))
        n0 = sum(s[1] == 0 for s in samples0)
        n1 = sum(s[1] == 1 for s in samples0)
        logging.info('classes: n0={} n1={}'.format(n0, n1))

        if len(samples0) > MAX_NB_SAMPLES:
            ri = np.random.permutation(range(len(samples0)))[:MAX_NB_SAMPLES]
            samples0 = [samples0[i] for i in ri]
            n0 = sum(s[1] == 0 for s in samples0)
            n1 = sum(s[1] == 1 for s in samples0)
            logging.info('{} samples left after truncation, n0={} n1={}'.format(len(samples0), n0, n1))

        samples = []
        all_pieces = set()
        for text, label in samples0:
            pieces = sent2pieces(splitter, tokenizer, text, max_text_len, params['repres'])
            samples.append((text, pieces, label))
            all_pieces.update(pieces)

        max_text_len += 2
        logging.info('max_text_len={}'.format(max_text_len))

        piece2index = {PAD_CHAR: 0}
        for piece in all_pieces:
            if piece != PAD_CHAR:
                piece2index[piece] = len(piece2index)

        nb_pieces = len(piece2index)
        computed_params['nb_pieces'] = nb_pieces
        computed_params['max_text_len'] = max_text_len
        logging.info('nb_pieces={}'.format(nb_pieces))

        logging.info('Vectorization of {} samples'.format(len(samples)))
        X_data = np.zeros((len(samples), max_text_len), dtype=np.int32)
        y_data = np.zeros(len(samples), dtype=np.bool)
        for isample, (text, pieces, label) in enumerate(samples):
            for i, piece in enumerate(pieces):
                X_data[isample, i] = piece2index[piece]
                y_data[isample] = label

        if pred_filepath:
            pred_samples = [s.strip() for s in io.open(pred_filepath, 'r', encoding='utf-8').readlines()]
            X_pred = np.zeros((len(pred_samples), max_text_len), dtype=np.int32)
            for isample, text in enumerate(pred_samples):
                pieces = sent2pieces(splitter, tokenizer, text.strip(), max_text_len, params['repres'])[:max_text_len]
                for i, piece in enumerate(pieces):
                    if piece in piece2index:
                        X_pred[isample, i] = piece2index[piece]

        del samples0
        del samples
    elif params['repres'] == 'words':
        max_text_len = 0

        w2v = dict()
        if params['w2v_name']:
            if params['w2v_name'] in w2v_storage:
                w2v = w2v_storage[params['w2v_name']]
            else:
                if is_linux():
                    w2v_path = os.path.expanduser('~/polygon/w2v/' + params['w2v_name'] + '.bin')
                else:
                    w2v_path = 'e:/polygon/w2v/' + params['w2v_name'] + '.bin'

                if params['w2v_name'].startswith('fasttext'):
                    w2v = gensim.models.wrappers.FastText.load_fasttext_format(w2v_path)
                else:
                    w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=not w2v_path.endswith('.txt'))
                w2v_storage[params['w2v_name']] = w2v

        samples = None
        if word_samples0 is not None:
            samples, max_text_len, nb_features, hash_params = word_samples0

            if params['use_postagger'] != hash_params['use_postagger'] or params['w2v_name'] != hash_params['w2v_name']:
                samples = None

        if samples is None:
            all_features = set([PAD_FEATURE])
            samples00 = []
            #for file in glob.glob(os.path.join('../data', '*.csv')):
            for file0 in data_files:
                file = os.path.join('../data', file0)
                if os.path.exists(file):
                    logging.info('Loading samples from {}...'.format(file))
                    df = pd.read_csv(file, encoding='utf-8', delimiter='\t', quoting=csv.QUOTE_MINIMAL)
                    logging.info('df.shape={}'.format(df.shape))
                    for irow, row in df.iterrows():
                        text = row['text']
                        if len(text) <= MAX_TEXT_LEN:
                            label = row['class']
                            tokens = tokenizer.tokenize(text)

                            if params['use_postagger']:
                                postags = tagger.tag(tokens)
                                features = [get_word_features(token, word_postags[1], word2tags, w2v, wc2v)
                                            for (token, word_postags)
                                            in zip(tokens, postags)]
                            else:
                                features = [get_word_features(token, u'', word2tags, w2v, wc2v) for token in tokens]

                                for fx in features:
                                    all_features.update(map(operator.itemgetter(0), fx))

                            samples00.append((text, features, label))
                            max_text_len = max(max_text_len, len(tokens))
                            if len(samples00) >= MAX_NB_SAMPLES:
                                break

                    del df

            logging.info('{} samples loaded'.format(len(samples00)))
            n0 = sum(s[2] == 0 for s in samples00)
            n1 = sum(s[2] == 1 for s in samples00)
            logging.info('classes: n0={} n1={}'.format(n0, n1))

            nb_features = len(all_features)
            feature2index = dict((f, i) for i, f in enumerate(all_features))
            logging.info('nb_features={}'.format(nb_features))

            if len(samples00) > MAX_NB_SAMPLES:
                #ri = np.random.permutation(range(len(samples00)))[:MAX_NB_SAMPLES]
                #samples00 = [samples00[i] for i in ri]
                samples00 = samples00[:MAX_NB_SAMPLES]
                n0 = sum(s[2] == 0 for s in samples00)
                n1 = sum(s[2] == 1 for s in samples00)
                logging.info('{} samples left after truncation, n0={} n1={}'.format(len(samples00), n0, n1))

            samples0 = []
            for text, feature_sets, label in samples00:
                feature_list = []

                for feature_set in feature_sets:
                    fx = np.zeros(nb_features, dtype=np.float32)
                    for feature_name, feature_value in feature_set:
                        fx[feature2index[feature_name]] = feature_value
                    feature_list.append(fx)

                # Добавляем заполнители
                n = max(0, max_text_len - len(feature_sets))
                for _ in range(n):
                    fx = np.zeros(nb_features, dtype=np.float32)
                    fx[feature2index[PAD_FEATURE]] = 1.0
                    feature_list.append(fx)

                samples0.append((text, feature_list, label))

            del samples00

            hash_params = dict()  # будем сбрасывать хэш загруженных сэмплов, если поменялись параметры фич
            hash_params['use_postagger'] = params['use_postagger']
            hash_params['w2v_name'] = params['w2v_name']
            word_samples0 = (samples0, max_text_len, nb_features, hash_params)
            samples = samples0

            logging.info('max_text_len={}'.format(max_text_len))

        computed_params['nb_features'] = nb_features
        computed_params['max_text_len'] = max_text_len

        logging.info('Vectorization of {} samples'.format(len(samples)))
        X_data = np.zeros((len(samples), max_text_len, nb_features), dtype=np.float32)
        y_data = np.zeros(len(samples), dtype=np.bool)

        for isample, (text, features_list, label) in enumerate(samples):
            y_data[isample] = label

            for iword, tok_features in enumerate(features_list):
                X_data[isample, iword, :] = tok_features

        if pred_filepath:
            pred_samples = [s.strip() for s in io.open(pred_filepath, 'r', encoding='utf-8').readlines()]
            X_pred = np.zeros((len(pred_samples), max_text_len, nb_features), dtype=np.float32)
            for isample, text in enumerate(pred_samples):
                tokens = tokenizer.tokenize(text.strip())[:max_text_len]

                if params['use_postagger']:
                    postags = tagger.tag(tokens)
                    feature_sets = [get_word_features(token, word_postags[1], word2tags, w2v, wc2v)
                                for (token, word_postags)
                                in zip(tokens, postags)]
                else:
                    feature_sets = [get_word_features(token, u'', word2tags, w2v, wc2v) for token in tokens]

                features_list = []

                for feature_set in feature_sets:
                    fx = np.zeros(nb_features, dtype=np.float32)
                    for feature_name, feature_value in feature_set:
                        if feature_name in feature2index:
                            fx[feature2index[feature_name]] = feature_value
                    features_list.append(fx)

                # Добавляем заполнители
                n = max(0, max_text_len - len(feature_sets))
                for _ in range(n):
                    fx = np.zeros(nb_features, dtype=np.float32)
                    fx[feature2index[PAD_FEATURE]] = 1.0
                    features_list.append(fx)

                for iword, tok_features in enumerate(features_list):
                    X_pred[isample, iword, :] = tok_features

    gc.collect()
    return computed_params, X_data, y_data, pred_samples, X_pred



def estimate(params, wc2v):
    computed_params, X_data, y_data, _, _ = prepare_data(params, pred_filepath=None, wc2v=wc2v)

    logging.info('Start training on folds with params={}'.format(get_params_str(params)))

    kfold_acc_sum = 0.0
    kfold_acc_denom = 0.0
    kfold_threshold_sum = 0.0
    kfold_accs = []

    kf = StratifiedKFold(n_splits=3)
    for ifold, (train_index, test_index) in enumerate(kf.split(X_data, y_data)):

        X_train, X_val = X_data[train_index], X_data[test_index]
        y_train, y_val = y_data[train_index], y_data[test_index]

        for probe in range(1):  # поставить >1, если будем усреднять несколько запусков нейросетки
            model = create_model(params, computed_params)
            train_model(params, model, X_train, y_train, X_val, y_val)
            y_pred = model.predict(X_val)
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_val, y_pred)
            threshold = thresholds[np.argmax(tpr - fpr)]
            kfold_threshold_sum += threshold

            y_pred2 = (y_pred >= threshold).astype(np.int)
            acc = sklearn.metrics.accuracy_score(y_true=y_val, y_pred=y_pred2)
            logging.info('KFOLD[{},{}]: threshold={:6.4f} accuracy={:6.4f}'.format(ifold, probe, threshold, acc))
            kfold_acc_sum += acc
            kfold_acc_denom += 1
            kfold_accs.append(acc)

    threshold = kfold_threshold_sum / kfold_acc_denom
    acc = kfold_acc_sum / kfold_acc_denom
    std = np.std(kfold_accs)

    return acc, std, threshold


ho_wc2v = None
cur_best_acc = 0.0
def ho_objective(space):
    global cur_best_acc

    params = dict(filter(lambda kv: kv[0] != 'net_arch_type', space.items()))
    params.update(space['net_arch_type'].items())

    acc, std, threshold = estimate(params, ho_wc2v)
    gc.collect()

    logging.info('Accuracy mean={:6.4f} std={:6.4f} threshold={:6.4f}'.format(acc, std, threshold))

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


def train_and_predict(params, input_file, output_file):
    logging.info('Start training predictor model with params={}'.format(get_params_str(params)))
    computed_params, X_data, y_data, pred_samples, X_pred = prepare_data(params, pred_filepath=input_file, wc2v=wc2v)
    model = create_model(params, computed_params)

    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=12345678)
    train_model(params, model, X_train, y_train, X_val, y_val)

    y_pred = model.predict(X_val)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_val, y_pred)
    threshold = thresholds[np.argmax(tpr - fpr)]

    y_pred2 = (y_pred >= threshold).astype(np.int)
    acc_val = sklearn.metrics.accuracy_score(y_true=y_val, y_pred=y_pred2)
    f1_val = sklearn.metrics.f1_score(y_true=y_val, y_pred=y_pred2)
    logging.info('threshold={} acc_val={} f1_val={}'.format(threshold, acc_val, f1_val))

    # Прогоняем разметочные данные через обученную модель
    logging.info('Computing predictions for {} samples'.format(len(pred_samples)))
    y_pred = model.predict(X_pred)

    #for isample in range(len(pred_samples)):
    #    print(u'DEBUG@598 isample={} X_pred[0:10]={} y={}'.format(isample, X_pred[isample, 0:10, :], y_pred[isample]))

    y_pred2 = (y_pred >= threshold).astype(np.int)

    logging.info('Writing {} predictions to "{}"'.format(len(pred_samples), output_file))
    res = pd.DataFrame(columns='text class'.split())
    res_rows = []
    for isample, (text, y) in enumerate(zip(pred_samples, y_pred2)):
        res_rows.append({'text': text, 'class': y[0]})
    res = res.append(res_rows, ignore_index=True)

    res.to_csv(output_file, sep='\t', encoding='utf-8', quoting=csv.QUOTE_MINIMAL, index=None)


FILLER_CHAR = u' '  # символ для выравнивания слов по одинаковой длине
BEG_CHAR = u'['  # символ отмечает начало цепочки символов слова
END_CHAR = u']'  # символ отмечает конец цепочки символов слова


def pad_word(word, max_word_len):
    return BEG_CHAR + word + END_CHAR + (max_word_len - len(word)) * FILLER_CHAR


def unpad_word(word):
    return word.strip()[1:-1]


def raw_wordset(wordset, max_word_len):
    return [(pad_word(word, max_word_len), pad_word(word, max_word_len)) for word in wordset]

missing_chars = set()
def vectorize_word(word, corrupt_word, X_batch, y_batch, irow, char2index):
    for ich, (ch, corrupt_ch) in enumerate(zip(word, corrupt_word)):
        if ich < X_batch.shape[1]:
            try:
                if corrupt_ch not in char2index:
                    if corrupt_ch not in missing_chars:
                        print(u'Char "{}" code={} word="{}" missing in char2index'.format(corrupt_ch, hex(ord(corrupt_ch)), corrupt_word.strip()))
                        missing_chars.add(corrupt_ch)
                else:
                    X_batch[irow, ich] = char2index[corrupt_ch]

                if ch not in char2index:
                    if ch not in missing_chars:
                        print(u'Char "{}" code={} word="{}" missing in char2index'.format(ch, hex(ord(ch)), word.strip()))
                        missing_chars.add(corrupt_ch)
                else:
                    y_batch[irow, ich, char2index[ch]] = True
            except UnicodeEncodeError:
                print(u'Char with code={} missing in char2index'.format(hex(ord(corrupt_ch))))


def build_test(wordset, max_word_len, char2index):
    ndata = len(wordset)
    nb_chars = len(char2index)
    X_data = np.zeros((ndata, max_word_len + 2), dtype=np.int32)
    y_data = np.zeros((ndata, max_word_len + 2, nb_chars), dtype=np.float32)

    for irow, (word, corrupt_word) in enumerate(wordset):
        vectorize_word(word, corrupt_word, X_data, y_data, irow, char2index)

    return X_data, y_data


def build_input(wordset, max_word_len, char2index):
    X, y_unused = build_test(wordset, max_word_len, char2index)
    return X


def vectorize_wc2v(output_words):
    wc2v = dict()

    with open(os.path.join(tmp_folder, 'wordchar2vector.config'), 'r') as f:
        model_config = json.load(f)

    # грузим готовую модель
    with open(model_config['arch_filepath'], 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(model_config['weights_path'])

    vec_size = model_config['vec_size']
    max_word_len = model_config['max_word_len']
    char2index = model_config['char2index']

    nb_words = len(output_words)

    wx = list(output_words)
    words = raw_wordset(wx, max_word_len)
    batch_size = 100

    words_remainder = nb_words
    word_index = 0
    while words_remainder > 0:
        print('words_remainder={:<10d}'.format(words_remainder), end='\r')
        nw = min(batch_size, words_remainder)
        batch_words = words[word_index:word_index + nw]
        X_data = build_input(batch_words, max_word_len, char2index)
        y_pred = model.predict(x=X_data, batch_size=nw, verbose=0)

        for iword, (word, corrupt_word) in enumerate(batch_words):
            word_vect = y_pred[iword, :]
            naked_word = unpad_word(word)
            assert (len(naked_word) > 0)
            wc2v[naked_word] = word_vect

        word_index += nw
        words_remainder -= nw

    return wc2v


# ---------------------------------------------------------------------------------------

if __name__ == '__main__':
    # Данные для разметки берутся из данного текстового файла.
    input_file = '../data/input_file.txt'

    # Результаты разметки данных из input_file будут сохранены в указанном файле
    output_file = os.path.join(tmp_folder, 'model1.result.csv')


    init_trainer_logging(os.path.join(tmp_folder, 'gridsearch_model1.log'))
    logging.info('Start')

    logging.info('Loading dictionaries...')
    tokenizer = TokenizerAGRR()
    tokenizer.load()

    word2tags = ruword2tags.RuWord2Tags()
    word2tags.load()

    tagger = rupostagger.RuPosTagger()
    tagger.load()

    all_words = set()
    #for file in glob.glob(os.path.join('../data', '*.csv')):
    for file0 in data_files:
        file = os.path.join('../data', file0)
        if os.path.exists(file):
            df = pd.read_csv(file, encoding='utf-8', delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            for text in df['text'].values:
                words = tokenizer.tokenize(text)
                all_words.update(words)

    if run_best:
        with io.open(input_file, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                words = tokenizer.tokenize(line.strip())
                all_words.update(words)

    logging.info('{} words for wc2v vectorization'.format(len(all_words)))
    wc2v = vectorize_wc2v(all_words)
    ho_wc2v = wc2v

    if do_gridsearch:
        logging.info('Building the grid...')
        # TODO: вынести в функцию-генератор, формировать лениво.
        grid = []
        params = dict()

        for optimizer in ['adam', 'nadam', 'ftml']:
            params['optimizer'] = optimizer

            for repres in ['words', 'sentencepiece','chars', 'syllables']:
                params['repres'] = repres

                if repres == 'words':
                    for w2v_name in ['w2v.CBOW=1_WIN=5_DIM=64', 'w2v.CBOW=0_WIN=5_DIM=64', '']:
                        params['w2v_name'] = w2v_name

                        for use_postagger in [False, True]:
                            params['use_postagger'] = use_postagger

                            for batch_size in [300, 200, 100]:
                                params['batch_size'] = batch_size

                                for units2 in [20, 15, 10, 5]:
                                    params['units2'] = units2

                                    for activation2 in ['sigmoid', 'relu']:  # 'pelu'
                                        params['activation2'] = activation2

                                        for normalization in ['', 'BatchNormalization', 'GroupNormalization', 'InstanceNormalization']:
                                            params['normalization'] = normalization

                                            for net_arch in ['lstm', 'cnn(1)', 'cnn(2)']:
                                                params['net_arch'] = net_arch

                                                if net_arch == 'lstm':
                                                    for units1 in [160, 128, 64]:
                                                        params['units1'] = units1

                                                        for dropout_rate in [0.6, 0.5, 0.4, 0.3]:
                                                            params['dropout_rate'] = dropout_rate

                                                            grid.append(params.copy())
                                                else:
                                                    for pooling in ['max']:  #, 'average']:
                                                        params['pooling'] = pooling

                                                        for nb_filters in [32, 64, 96]:
                                                            params['nb_filters'] = nb_filters

                                                            for min_kernel_size in [1]:
                                                                params['min_kernel_size'] = min_kernel_size

                                                                for max_kernel_size in [2, 3]:
                                                                    params['max_kernel_size'] = max_kernel_size

                                                                    for dropout_rate in [0.0, 0.1, 0.2, 0.3]:
                                                                        params['dropout_rate'] = dropout_rate

                                                                        grid.append(params.copy())

                elif repres == 'sentencepiece':
                    for spm_items in [30000, 20000, 10000]:
                        params['spm_items'] = spm_items

                        for piece_dim in [-1, 80, 64]:  # -1 означает единичную диагональ без тренировки
                            params['piece_dim'] = piece_dim

                            for batch_size in [150, 100]:
                                params['batch_size'] = batch_size

                                for units2 in [5, 10]:
                                    params['units2'] = units2

                                    for activation2 in ['sigmoid', 'relu']:  # 'pelu'
                                        params['activation2'] = activation2

                                        for normalization in ['', 'BatchNormalization', 'GroupNormalization', 'InstanceNormalization']:
                                            params['normalization'] = normalization

                                            for net_arch in ['cnn(2)', 'cnn(1)']:
                                                params['net_arch'] = net_arch

                                                for pooling in ['max', 'average']:
                                                    params['pooling'] = pooling

                                                    for nb_filters in [64, 96]:
                                                        params['nb_filters'] = nb_filters

                                                        for min_kernel_size in [2, 1]:
                                                            params['min_kernel_size'] = min_kernel_size

                                                            for max_kernel_size in [6, 5]:
                                                                params['max_kernel_size'] = max_kernel_size

                                                                for dropout_rate in [0.0, 0.1, 0.2]:
                                                                    params['dropout_rate'] = dropout_rate

                                                                    grid.append(params.copy())

                elif repres in ['syllables', 'chars']:
                    for piece_dim in [-1, 80, 64]:  # -1 означает единичную диагональ без тренировки
                        params['piece_dim'] = piece_dim

                        for batch_size in [150, 100]:
                            params['batch_size'] = batch_size

                            for units2 in [5, 10]:
                                params['units2'] = units2

                                for activation2 in ['sigmoid', 'relu']:  # 'pelu'
                                    params['activation2'] = activation2

                                    for normalization in ['BatchNormalization', '', 'GroupNormalization',
                                                          'InstanceNormalization']:
                                        params['normalization'] = normalization

                                        for net_arch in ['cnn(2)', 'cnn(1)']:
                                            params['net_arch'] = net_arch

                                            for pooling in ['max', 'average']:
                                                params['pooling'] = pooling

                                                for nb_filters in [64, 96]:
                                                    params['nb_filters'] = nb_filters

                                                    for min_kernel_size in [2, 1]:
                                                        params['min_kernel_size'] = min_kernel_size

                                                        for max_kernel_size in [6, 5]:
                                                            params['max_kernel_size'] = max_kernel_size

                                                            for dropout_rate in [0.0, 0.1, 0.2]:
                                                                params['dropout_rate'] = dropout_rate

                                                                grid.append(params.copy())

        logging.info('Start grid search over {} parameter sets'.format(len(grid)))
        best_params = None
        best_score = 0.0
        best_std = 0.0
        wrt_best = io.open(os.path.join(tmp_folder, 'gridsearch_model1.best.txt'), 'w')

        for iparams, params in enumerate(grid):
            logging.info('Parameter set #{}/{}: {}'.format(iparams, len(grid), get_params_str(params)))
            logging.info('Current best_score={}'.format(best_score))

            acc, std, threshold = estimate(params, wc2v)
            gc.collect()

            logging.info('Accuracy mean={:6.4f} std={:6.4f} threshold={:6.4f}'.format(acc, std, threshold))

            if acc > best_score:
                logging.info('!!! New best score={} for params={}'.format(acc, get_params_str(params)))
                best_params = params
                best_params['threshold'] = threshold
                best_score = acc
                best_std = std

                with open(best_params_path, 'w') as f:
                    json.dump(best_params, f)

                wrt_best.write(u'\n\n{}\nAccuracy mean={:6.4f} std={:6.4f}\n'.format(get_params_str(best_params), best_score, best_std))
                wrt_best.flush()


        wrt_best.close()
        logging.info('Gridsearch complete.')

    if do_hyperopt:
        wrt_best = io.open(os.path.join(tmp_folder, 'gridsearch_model1.best.txt'), 'w')

        space = hp.choice('repres_type', [
            {
                'repres': 'words',

                'optimizer': hp.choice('optimizer', ['adam', 'nadam', 'ftml']),
                'batch_size': hp.choice('batch_size', range(100, 400, 50)),
                'w2v_name': hp.choice('w2v_name', ['w2v.CBOW=1_WIN=5_DIM=64']),  # 'w2v.CBOW=1_WIN=5_DIM=64' 'fasttext.CBOW=1_WIN=5_DIM=64'
                'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.6),
                'use_postagger': hp.choice('use_postagger', [False]),
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
                    #{'net_arch': 'cnn(2)',
                    # 'pooling': hp.choice('pooling_cnn2', ['max']),
                    # 'nb_filters': hp.choice('nb_filters_cnn2', [32, 64, 96]),
                    # 'min_kernel_size': hp.choice('min_kernel_size_cnn2', [1]),
                    # 'max_kernel_size': hp.choice('max_kernel_size_cnn2', [2, 3, 4]),
                    # },
                ]),
            },
            # todo - добавить подпространства параметров для остальных вариантов 'repres' из gridsearch'а
        ])

        trials = Trials()
        best = hyperopt.fmin(fn=ho_objective,
                             space=space,
                             algo=HYPEROPT_ALGO,
                             max_evals=100,
                             trials=trials,
                             verbose=1)

    if run_best:
        logging.info('Running the best model')

        with open(best_params_path, 'r') as f:
            best_params = json.load(f)

        logging.info('best_params={}'.format(get_params_str(best_params)))

        word_samples0 = None

        train_and_predict(best_params, input_file, output_file)

        logging.info('End of running the best model.')
