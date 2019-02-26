# -*- coding: utf-8 -*-
"""
Grid search для подбора параметров модели, маркирующей слова флагами
принадлежности в диапазоны cV, cR1, cR2, R1, R2

Используется keras_contrib.layers.CRF и seq2seq модели

Также - разметка тестовых данных с помощью модели с оптимальными параметрами.
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import os
import io
import re
#import glob
import gc
import operator
import numpy as np
import pandas as pd
import csv
import json

import logging
import logging.handlers

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


import gensim
from gensim.models.wrappers import FastText

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input
from keras.layers import recurrent
from keras.layers.core import RepeatVector, Dense
from keras.layers.wrappers import Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.models import model_from_json

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
import keras_contrib.optimizers
from keras_contrib.optimizers import FTML

import hyperopt
from hyperopt import hp, tpe, STATUS_OK, Trials

import ruword2tags
import rusyllab
import rutokenizer

from agrr_tokenizer import TokenizerAGRR


MAX_TEXT_LEN = 200  # макс. число элементов в тексте; если больше - не используем для обучения
MAX_NB_SAMPLES = 25000
data_files = ['train.csv', 'dev.csv']


# Выполнить поиск оптимальных параметров
do_hyperopt = False  # Выполнять подбор с помощью hyperopt
do_gridsearch = False  # Выполнять подбор с помощью gridsearch

# Взять лучшие параметры (из сценария do_gridsearch), создать модель, обучить ее,
# обработать входной файл, сохранить результаты в csv.
run_best = True

# Модель для метки V обучается отдельно от модели для меток cV, cR1, cR2, R1, R2
#labels_group = 'cV,cR1,cR2,R1,R2'  # 'V' или 'cV,cR1,cR2,R1,R2'
labels_group = 'V'

tmp_folder = '../tmp'

PAD_WORD = u''
PAD_FEATURE = u'<padding>'
NULL_LABEL = u'<null>'


# алгоритм сэмплирования гиперпараметров
HYPEROPT_ALGO = tpe.suggest  # tpe.suggest OR hyperopt.rand.suggest


def get_params_str(params):
    return u' '.join('{}={}'.format(p, v) for (p, v) in params.items())


def init_trainer_logging(logfile_path):
    # настраиваем логирование в файл и эхо-печать в консоль
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    lf = logging.FileHandler(logfile_path, mode='w')

    lf.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    lf.setFormatter(formatter)
    logging.getLogger('').addHandler(lf)


def is_num(token):
    return re.match('^[0-9]+$', token)


def is_punkt(word):
    return word[0] in u'‼≠™®•·[¡+<>`~;.,‚?!-…№”“„{}|‹›/\'"–—_‑:«»*]()‘’≈'


missing_w2v_words = set()
def get_word_features(word, postags, word2tags, w2v, wc2v):
    features = set()

    if is_num(word):
        features.add(('<number>', 1.0))
    elif is_punkt(word[0]):
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

        if lword not in missing_w2v_words:
            missing_w2v_words.add(lword)
            if not is_num(lword):
                logging.warning(u'Word "{}" is missing in w2v'.format(lword))

    if True:
        if word in wc2v:
            v = wc2v[word]
            for ix, x in enumerate(v):
                features.add((u'wc2v[{}]'.format(ix), x))

    if postags:
        for tag in postags.split('|'):
            features.add((tag, 1.0))

    return list(features)


def vectorize_data(samples, pred_samples, tokenizer, w2v, wc2v, word2tags, params):
    samples2 = []
    all_features = set([PAD_FEATURE])
    all_labels = set()
    max_text_len = max(len(sample[1]) for sample in samples)

    for sample in samples:
        sample2 = []
        tokens = sample[1]
        for iword, token in enumerate(tokens):
            word = token[0]
            label = token[1]
            all_labels.add(label)
            features = get_word_features(word, u'', word2tags, w2v, wc2v)
            all_features.update(map(operator.itemgetter(0), features))
            sample2.append((word, features, label))

        # выравниваем пустышками справа
        npad = max(0, max_text_len-len(tokens))
        for _ in range(npad):
            sample2.append((PAD_WORD, [(PAD_FEATURE, 1.0)], NULL_LABEL))

        samples2.append(sample2)

    nb_features = len(all_features)

    label2index = dict((l, i) for (i, l) in enumerate(all_labels))
    feature2index = dict((f, i) for (i, f) in enumerate(all_features))

    nb_labels = len(all_labels)
    computed_params = {'nb_features': nb_features,
                       'max_text_len': max_text_len,
                       'nb_labels': nb_labels,
                       'label2index': label2index,
                       'feature2index': feature2index
                       }

    nb_samples = len(samples2)
    X_data = np.zeros((nb_samples, max_text_len, nb_features), dtype=np.float32)
    y_data = np.zeros((nb_samples, max_text_len, nb_labels), dtype=np.bool)

    for isample, sample in enumerate(samples2):
        for iword, (word, features, label) in enumerate(sample):
            y_data[isample, iword, label2index[label]] = 1
            for f, v in features:
                X_data[isample, iword, feature2index[f]] = v

    X_pred = None
    if pred_samples:
        X_pred = np.zeros((len(pred_samples), max_text_len, nb_features), dtype=np.float32)
        for isample, sample in enumerate(pred_samples):
            sample2 = []
            tokens = tokenizer.tokenize(sample)[:max_text_len]
            for iword, word in enumerate(tokens):
                features = get_word_features(word, u'', word2tags, w2v, wc2v)
                sample2.append((word, features))

            # выравниваем пустышками справа
            npad = max(0, max_text_len-len(tokens))
            for _ in range(npad):
                sample2.append((PAD_WORD, [(PAD_FEATURE, 1.0)]))

            for iword, (word, features) in enumerate(sample2):
                for f, v in features:
                    if f in feature2index:
                        X_pred[isample, iword, feature2index[f]] = v

    return X_data, y_data, X_pred, computed_params


def create_model(params, computed_params):
    input = Input(shape=(computed_params['max_text_len'],
                         computed_params['nb_features'],),
                  dtype='float32', name='input')

    if params['optimizer'] == 'ftml':
        opt = keras_contrib.optimizers.FTML()
    else:
        opt = params['optimizer']

    if params['net_arch'] == 'crf':
        net = input

        for _ in range(params['nb_rnn']):
            net = Bidirectional(recurrent.LSTM(units=params['rnn_units1'],
                                               dropout=params['dropout_rate'],
                                               return_sequences=True))(net)

        net = CRF(units=computed_params['nb_labels'], sparse_target=False)(net)
        model = Model(inputs=[input], outputs=net)
        model.compile(loss=crf_loss, optimizer=opt, metrics=[crf_viterbi_accuracy])
        model.summary()
    elif params['net_arch'] == 'lstm':
        net = Bidirectional(recurrent.LSTM(units=params['rnn_units1'],
                                           dropout=params['dropout_rate'],
                                           return_sequences=False))(input)

        for _ in range(params['nb_dense1']):
            net = Dense(units=params['rnn_units1'], activation=params['activation1'])(net)

        decoder = RepeatVector(computed_params['max_text_len'])(net)
        decoder = recurrent.LSTM(params['rnn_units2'], return_sequences=True)(decoder)
        decoder = TimeDistributed(Dense(units=computed_params['nb_labels'],
                                        activation='softmax'), name='output')(decoder)

        model = Model(inputs=[input], outputs=decoder)

        model.compile(loss='categorical_crossentropy', optimizer=opt)
        model.summary()

    return model


def train_model(model, X_train, y_train, X_val, y_val, params, computed_params):
    weights_path = os.path.join(tmp_folder, 'model2.kerascrf.weights')
    if params['net_arch'] == 'crf':
        monitor_metric = 'val_crf_viterbi_accuracy'
    else:
        monitor_metric = 'val_loss'

    model_checkpoint = ModelCheckpoint(weights_path, monitor=monitor_metric,
                                       verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')

    callbacks = [model_checkpoint, early_stopping]

    if True:  # ДЛЯ ОТЛАДКИ!!!
        model.fit(x=X_train, y=y_train,
                  validation_data=(X_val, y_val),
                  epochs=1000,
                  batch_size=params['batch_size'],
                  verbose=2,
                  callbacks=callbacks)
    else:
        print(u'------ DEBUG@284 УБРАТЬ ОТЛАДКУ!!! ----------')

    model.load_weights(weights_path)

    y_pred = model.predict(X_val)

    # оценка точности per instance
    nb_total = len(y_val)
    nb_good = sum(np.array_equal(y_pred[i, :], y_val[i, :]) for i in range(nb_total))
    acc_perinstance = nb_good / float(nb_total)

    # оценка точности per token
    # (!) получается слишком оптимистичная оценка из-за большого количества заполнителей,
    # отображаемых на NULL_LABEL.
    yy_pred = np.argmax(y_pred, axis=-1)
    yy_val = np.argmax(y_val,  axis=-1)
    nb_total = yy_pred.shape[0] * yy_pred.shape[1]
    nb_good = 0
    for irow in range(yy_pred.shape[0]):
        n1 = np.sum(np.equal(yy_pred[irow], yy_val[irow]))
        nb_good += n1
    acc_pertoken = nb_good / float(nb_total)

    return acc_pertoken, acc_perinstance


def estimate(samples, params, tokenizer, w2v, wc2v, word2tags):
    kfold_acc_sum = 0.0
    kfold_acc_denom = 0.0
    kfold_acc = []

    logging.info('Start training with params={}'.format(get_params_str(params)))
    kf = KFold(n_splits=3)
    for ifold, (train_index, test_index) in enumerate(kf.split(samples)):
        # TODO - можно еще делать по несколько повторов на каждый фолд, чтобы усреднить результаты сеток.
        X_data, y_data, X_pred, computed_params = vectorize_data(samples, None, tokenizer, w2v, wc2v, word2tags, params)
        X_train, y_train = X_data[train_index], y_data[train_index]
        X_val, y_val = X_data[test_index], y_data[test_index]

        model = create_model(params, computed_params)
        acc_pertoken, acc_perinstance = train_model(model, X_train, y_train, X_val, y_val, params, computed_params)
        logging.info('KFOLD[{}] accuracy --> per_token={:6.4f} per_instance={:6.4f}'.format(ifold, acc_pertoken, acc_perinstance))

        kfold_acc_sum += acc_pertoken
        kfold_acc_denom += 1.0
        kfold_acc.append(acc_pertoken)

    acc = kfold_acc_sum / kfold_acc_denom
    std = np.std(kfold_acc)

    return acc, std


ho_word2tags = None
ho_tokenizer = None
ho_w2v = None
ho_wc2v = None
ho_samples = None
cur_best_acc = 0.0
def ho_objective(space):
    global cur_best_acc

    params = dict(filter(lambda kv: kv[0] != 'net_arch_type', space.items()))
    params.update(space['net_arch_type'].items())

    acc, std = estimate(ho_samples, params, ho_tokenizer, ho_w2v, ho_wc2v, ho_word2tags)
    gc.collect()

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



def split_range(s):
    if isinstance(s, float) and np.isnan(s):
        return []
    else:
        return [tuple(map(int, r.split(':'))) for r in s.split(' ')]


def find_label_sequences(label_index, labels, tokens, labels_group):
    """
    Среди токенов tokens и их меток labels ищем все непрерывные цепочки метки с кодом label_index.
    Возвращается строка со списком пар посимвольных позиций границ этих цепочек вида 10:15 24:32
    """
    ranges = []

    n = min(len(labels), len(tokens))
    i = 0

    while i < n:
        if labels[i] == label_index:
            # Нашли начало цепочки токенов, помеченных нужной нам меткой
            start_tok_pos = i
            end_tok_pos = -1

            # Ищем конец цепочки
            for j in range(i, n):
                if labels[j] == label_index:
                    end_tok_pos = j
                    i += 1
                else:
                    break

            # Найдена очередная цепочка токенов, длиной минимум 1 токен.
            start_pos = tokens[start_tok_pos][1]  # начальная символьная позиция
            if labels_group == 'V':
                end_pos = start_pos  # для меток V конечная позиция всегда совпадает с начальной
            else:
                end_pos = tokens[end_tok_pos][2]  # конечная символьная позиция

            ranges.append('{}:{}'.format(start_pos, end_pos))

        i += 1

    return u' '.join(ranges)




FILLER_CHAR = u' '  # символ для выравнивания слов по одинаковой длине
BEG_CHAR = u'['  # символ отмечает начало цепочки символов слова
END_CHAR = u']'  # символ отмечает конец цепочки символов слова


def pad_word(word, max_word_len):
    return BEG_CHAR + word + END_CHAR + (max_word_len - len(word)) * FILLER_CHAR


def unpad_word(word):
    return word.strip()[1:-1]


def raw_wordset(wordset, max_word_len):
    return [(pad_word(word, max_word_len), pad_word(word, max_word_len)) for word in wordset]


def vectorize_word(word, corrupt_word, X_batch, y_batch, irow, char2index):
    for ich, (ch, corrupt_ch) in enumerate(zip(word, corrupt_word)):
        if corrupt_ch not in char2index:
            print(u'Char "{}" code={} word="{}" missing in char2index'.format(corrupt_ch, ord(corrupt_ch), corrupt_word.strip()))
        else:
            X_batch[irow, ich] = char2index[corrupt_ch]
        if ch not in char2index:
            print(u'Char "{}" code={} word="{}" missing in char2index'.format(ch, ord(ch), word.strip()))
        else:
            y_batch[irow, ich, char2index[ch]] = True


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




if __name__ == '__main__':
    init_trainer_logging(os.path.join(tmp_folder, 'gridsearch_model2_kerascrf.log'))

    # Данные для разметки берутся из данного текстового файла.
    input_file = '../data/input_file.txt'

    best_params_path = os.path.join(tmp_folder, 'gridsearch_model2_kerascrf.{}.best_params.json'.format(labels_group))

    # Результаты разметки данных из input_file будут сохранены в указанном файле
    output_file = os.path.join(tmp_folder, 'model2.{}.result.csv'.format(labels_group))

    logging.info('Start')

    word2tags = ruword2tags.RuWord2Tags()
    word2tags.load()

    tokenizer = TokenizerAGRR()
    tokenizer.load()

    logging.info('Loading samples for training and validation...')
    samples = []

    nb_markup_errors = 0
    nb_markup_fixes = 0

    all_words = set()

    #for file in glob.glob(os.path.join('../data', '*.csv')):
    for file0 in data_files:
        file = os.path.join('../data', file0)
        if os.path.exists(file):
            logging.info('Loading samples from {}...'.format(file))
            df = pd.read_csv(file, encoding='utf-8', delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            for irow, row in df[df['class'] == 1].iterrows():
                try:
                    text = row['text']

                    cV = split_range(row['cV'])
                    cR1 = split_range(row['cR1'])
                    cR2 = split_range(row['cR2'])
                    V = split_range(row['V'])
                    R1 = split_range(row['R1'])
                    R2 = split_range(row['R2'])

                    tokens = tokenizer.tokenize2(text)
                    if len(text) <= MAX_TEXT_LEN:
                        start2iword = dict()
                        end2iword = dict()

                        for itoken, (token, start, end) in enumerate(tokens):
                            start2iword[start] = itoken
                            end2iword[end] = itoken

                        words = [(word, start, end, []) for (word, start, end) in tokens]

                        all_words.update(word for (word, _, _) in tokens)

                        if labels_group == 'V':
                            labels_list = [('V', V)]
                        else:
                            labels_list = [('cV', cV), ('cR1', cR1), ('cR2', cR2), ('R1', R1), ('R2', R2)]

                        for label, ranges in labels_list:
                            for start, end in ranges:
                                old_start = start
                                old_end = end

                                if start not in start2iword:
                                    if start - 1 in start2iword:
                                        start = start - 1
                                        end = end - 1
                                        nb_markup_fixes += 1
                                    elif start - 2 in start2iword:
                                        start = start - 2
                                        end = end - 2
                                        nb_markup_fixes += 1
                                    else:
                                        nb_markup_errors += 1
                                        raise Exception(u'Could not fix start position={} in text {}'.format(start, text))

                                if labels_group != 'V':
                                    if end not in end2iword:
                                        nb_markup_errors += 1
                                        raise Exception(u'Could not fix end position={} in text {}'.format(end, text))

                                if old_start != start or old_end != end:
                                    print(
                                        u'Start position corrected for label {} in text {}:\nold={} new={}\nold slice=[{}] new slice=[{}]\n'.format(
                                            label, text, old_start, start, text[old_start: old_end], text[start: end]))

                                if labels_group == 'V':
                                    words[start2iword[start]][3].append(label)
                                else:
                                    for iword in range(start2iword[start], end2iword[end] + 1):
                                        words[iword][3].append(label)

                        words2 = []
                        for word in words:
                            if len(word[3]) > 1:
                                raise Exception(u'Word={} in text={} got {} labels'.format(word[0], text, len(word[3])))
                            elif len(word[3]) == 1:
                                words2.append((word[0], word[3][0]))  # оставляем только одну метку вместо списка
                            elif len(word[3]) == 0:
                                words2.append((word[0],
                                               NULL_LABEL))  # помечаем прочие слова, не относящиеся к интересующим категориям

                        samples.append((text, words2))

                except Exception as e:
                    logging.error(u'Error {} for line {}'.format(e, text))

    if len(samples) > MAX_NB_SAMPLES:
        ri = np.random.permutation(range(len(samples)))[:MAX_NB_SAMPLES]
        samples = [samples[i] for i in ri]

    logging.info('nb_markup_fixes={}'.format(nb_markup_fixes))
    logging.info('nb_markup_errors={}'.format(nb_markup_errors))
    logging.info('samples count={}'.format(len(samples)))

    if run_best:
        with io.open(input_file, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                words = tokenizer.tokenize(line.strip())
                all_words.update(words)

    wc2v = vectorize_wc2v(all_words)


    if do_gridsearch:
        wrt_best = io.open(os.path.join(tmp_folder, 'gridsearch_model2_kerascrf.{}.best.txt'.format(labels_group)), 'w', encoding='utf-8')
        best_params = None
        best_score = 0

        params = dict()

        for w2v_name in ['w2v.CBOW=1_WIN=5_DIM=64.bin', 'w2v.CBOW=0_WIN=5_DIM=64.bin']:
            params['w2v_name'] = w2v_name

            w2v = dict()
            if w2v_name:
                w2v_path = os.path.expanduser('~/polygon/w2v/' + params['w2v_name'])
                if w2v_name.startswith('fasttext'):
                    w2v = FastText.load_fasttext_format(w2v_path)
                else:
                    w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=not w2v_path.endswith('.txt'))
            else:
                w2v = dict()

            for batch_size in [200, 150]:
                params['batch_size'] = batch_size

                for net_arch in ['crf']: # ['lstm']:  # 'crf'
                    params['net_arch'] = net_arch

                    if net_arch == 'lstm':
                        for rnn_units1 in [128, 100]:
                            params['rnn_units1'] = rnn_units1

                            for rnn_units2 in [100]:
                                params['rnn_units2'] = rnn_units2

                                for dropout_rate in [0.0]:
                                    params['dropout_rate'] = dropout_rate

                                    for nb_dense1 in [1]:
                                        params['nb_dense1'] = nb_dense1

                                        for activation1 in ['sigmoid', 'relu']:
                                            params['activation1'] = activation1

                                            acc, std = estimate(samples, params, tokenizer, w2v, word2tags)
                                            logging.info('accuracy={:6.4f} std={:6.4f}'.format(acc, std))

                                            if acc > best_score:
                                                logging.info('!!! New best score={} for params={}'.format(acc, get_params_str(params)))
                                                best_params = params
                                                best_score = acc

                                                with open(best_params_path, 'w') as f:
                                                    json.dump(best_params, f)

                                                wrt_best.write(u'\n\n{}\nscore={}\n'.format(get_params_str(best_params), best_score))
                                                wrt_best.flush()

                    elif net_arch == 'crf':
                        for nb_rnn in [1]:
                            params['nb_rnn'] = nb_rnn

                            for rnn_units1 in [100, 128, 160]:
                                params['rnn_units1'] = rnn_units1

                                for dropout_rate in [0.2, 0.1, 0.0, 0.3]:
                                    params['dropout_rate'] = dropout_rate

                                    acc, std = estimate(samples, params, tokenizer, w2v, wc2v, word2tags)
                                    logging.info('accuracy={} std={}'.format(acc, std))

                                    if acc > best_score:
                                        logging.info('!!! New best score={} for params={}'.format(acc, get_params_str(params)))
                                        best_params = params
                                        best_score = acc

                                        with open(best_params_path, 'w') as f:
                                            json.dump(best_params, f)

                                        wrt_best.write(u'\n\n{}\nscore={}\n'.format(get_params_str(best_params), best_score))
                                        wrt_best.flush()

        wrt_best.close()

        logging.info('Final best_params={}'.format(get_params_str(best_params)))
        logging.info('Final best_score={}'.format(best_score))

    if do_hyperopt:
        wrt_best = io.open(os.path.join(tmp_folder, 'gridsearch_model2_kerascrf.{}.best.txt'.format(labels_group)), 'w', encoding='utf-8')

        space = hp.choice('repres_type', [
            {
                'optimizer': hp.choice('optimizer', ['adam', 'nadam', 'ftml']),
                'batch_size': hp.choice('batch_size', range(100, 400, 50)),
                'w2v_name': hp.choice('w2v_name', ['w2v.CBOW=1_WIN=5_DIM=64.bin']),  # 'w2v.CBOW=1_WIN=5_DIM=64.bin', 'w2v.CBOW=0_WIN=5_DIM=64.bin', 'fasttext.CBOW=1_WIN=5_DIM=64.bin'
                'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5),
                'use_postagger': hp.choice('use_postagger', [False]),

                'net_arch_type': hp.choice('net_arch_type', [
                    #{'net_arch': 'lstm',
                    # 'rnn_units1': hp.choice('rnn_units1', range(64, 256)),
                    # 'rnn_units2': hp.choice('rnn_units2', range(64, 256)),
                    # 'nb_dense1': hp.choice('nb_dense1', [1]),
                    # 'activation1': hp.choice('activation1', ['sigmoid', 'relu']),
                    # },
                    {'net_arch': 'crf',
                     'rnn_units1': hp.choice('rnn_units1_crf', range(64, 256)),
                     'nb_rnn': hp.choice('nb_rnn', [1]),
                     },
                ])
            }
        ])

        ho_word2tags = ruword2tags.RuWord2Tags()
        ho_word2tags.load()

        ho_tokenizer = TokenizerAGRR()
        ho_tokenizer.load()

        ho_samples = samples

        w2v_name = 'w2v.CBOW=1_WIN=5_DIM=64.bin'
        w2v_path = os.path.expanduser('~/polygon/w2v/' + w2v_name)
        if w2v_name.startswith('fasttext'):
            ho_w2v = FastText.load_fasttext_format(w2v_path)
        else:
            ho_w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=not w2v_path.endswith('.txt'))

        ho_wc2v = wc2v

        trials = Trials()
        best = hyperopt.fmin(fn=ho_objective,
                             space=space,
                             algo=HYPEROPT_ALGO,
                             max_evals=100,
                             trials=trials,
                             verbose=1)

    if run_best:
        logging.info('Running the best model for {}'.format(labels_group))

        with open(best_params_path, 'r') as f:
            best_params = json.load(f)

        logging.info('best_params={}'.format(get_params_str(best_params)))

        pred_samples = [s.strip() for s in io.open(input_file, 'r', encoding='utf-8').readlines()]
        logging.info('pred_samples.count={}'.format(len(pred_samples)))

        w2v_name = best_params['w2v_name']
        w2v = dict()
        if w2v_name:
            w2v_path = os.path.expanduser('~/polygon/w2v/' + w2v_name)
            if w2v_name.startswith('fasttext'):
                w2v = FastText.load_fasttext_format(w2v_path)
            else:
                w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=not w2v_path.endswith('.txt'))
        else:
            w2v = dict()

        X_data, y_data, X_pred, computed_params = vectorize_data(samples, pred_samples, tokenizer, w2v, wc2v, word2tags, best_params)
        logging.debug('max_text_len={}'.format(computed_params['max_text_len']))
        X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=12345678)

        model = create_model(best_params, computed_params)

        acc_pertoken, acc_perinstance = train_model(model, X_train, y_train, X_val, y_val, best_params, computed_params)

        y_pred = model.predict(X_pred)
        y_pred = np.argmax(y_pred, axis=-1)
        logging.debug('y_pred.shape={}'.format(y_pred.shape))

        logging.info('Writing {} predictions to "{}"'.format(len(pred_samples), output_file))

        if labels_group == 'V':
            res = pd.DataFrame(columns='text V'.split())
        else:
            res = pd.DataFrame(columns='text cV cR1 cR2 R1 R2'.split())

        res_rows = []
        for isample, (text, y) in enumerate(zip(pred_samples, y_pred)):
            tokens = tokenizer.tokenize2(text)

            if labels_group == 'V':
                V = find_label_sequences(computed_params['label2index']['V'], y, tokens, labels_group)
                res_rows.append({'text': text, 'V': V})
            else:
                cV = find_label_sequences(computed_params['label2index']['cV'], y, tokens, labels_group)
                cR1 = find_label_sequences(computed_params['label2index']['cR1'], y, tokens, labels_group)
                cR2 = find_label_sequences(computed_params['label2index']['cR2'], y, tokens, labels_group)
                R1 = find_label_sequences(computed_params['label2index']['R1'], y, tokens, labels_group)
                R2 = find_label_sequences(computed_params['label2index']['R2'], y, tokens, labels_group)
                res_rows.append({'text': text, 'cV': cV, 'cR1': cR1, 'cR2': cR2, 'R1': R1, 'R2': R2})

        res = res.append(res_rows, ignore_index=True)

        res.to_csv(output_file, sep='\t', encoding='utf-8', quoting=csv.QUOTE_MINIMAL, index=None)

        logging.info('End of running the best model.')
