# -*- coding: utf-8 -*-
"""
Тренировка модели для превращения символьной цепочки слова в вектор - посимвольная векторная
аппроксимация морфологии.

RNN и CNN варианты энкодера. Реализовано на Keras.

Результат работы программы - файлы в каталоге ../tmp при обучении (--train 1)
1) wordchar2vector.arch - сохраненная архитектура нейросетки для ее восстановления с помощью
 model_from_json
2) wordchar2vector.model - веса оптимального варианта модели, сохраненные в одном из чекпоинтов
в ходе обучения.
3) wordchar2vector.config - JSON конфиг модели с основными параметрами, необходимыми для
подготовки входных векторов фич при использовании модели.
4) wordchar2vector.arch.png - отрисованная архитектура нейросетки
5) learning_curve__*.csv - данные по кривой обучения: точность модели по мере накопления эпох.

В ходе генерации векторов для списка слов по обученной модели (--vectorize 1) генерируются файлы:
1) wordchar2vector.dat - текстовый файл в формате, совместимом с w2v, содержащий слова и сгенерированные
нейросеткой их векторы.

Прочие подробности: https://github.com/Koziev/chatbot/blob/master/PyModels/trainers/README.wordchar2vector.md
"""

from __future__ import print_function

import argparse
import os
import logging
import pandas as pd
import glob
import io
import csv

import rutokenizer  # https://github.com/Koziev/rutokenizer

from wordchar2vector_trainer import Wordchar2Vector_Trainer
import logging_helpers
import console_helpers


# Здесь сохраним список слов, на которых тренируемся.
input_path = '../tmp/known_words.txt'

# Соберем слова для обучения и векторизации, взяв их из данных для обучения AGRR
tokenizer = rutokenizer.Tokenizer()
tokenizer.load()

known_words = set()
for file in glob.glob(os.path.join('../data', '*.csv')):
    logging.info(u'Loading samples from {}...'.format(file))
    df = pd.read_csv(file, encoding='utf-8', delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for irow, row in df.iterrows():
        text = row['text']
        words = tokenizer.tokenize(text)
        known_words.update(words)

# Добавим известные русские словоформы (см. https://github.com/Koziev/GrammarEngine)
with io.open('../data/wordforms.txt', 'r', encoding='utf-8') as rdr:
    for line in rdr:
        known_words.add(line.strip())

print('known_words.count={}'.format(len(known_words)))
with io.open(input_path, 'w', encoding='utf-8') as wrt:
    for word in known_words:
        wrt.write(u'{}\n'.format(word))

parser = argparse.ArgumentParser(description='Training the wordchar2vector embeddings for character surface of words')
parser.add_argument('--out_file', default='../tmp/wordchar2vector.dat', help='output text file containing with word vectors in word2vec text format')
parser.add_argument('--model_dir', help='folder with model files', default='../tmp')
parser.add_argument('--tmp_dir', default='../tmp', help='folder for learning history data')
parser.add_argument('--train', default=1, type=int)
parser.add_argument('--vectorize', default=0, type=int)
parser.add_argument('--dims', default=56, type=int)
parser.add_argument('--char_dims', default=0, type=int)
parser.add_argument('--tunable_char_embeddings', default=0, type=int)
parser.add_argument('--arch_type', default='gru(cnn)', choices='cnn rnn bidir_lstm lstm(lstm) lstm+cnn lstm(cnn) gru(cnn)'.split(), type=str)
parser.add_argument('--batch_size', default=350, type=int)
parser.add_argument('--min_batch_size', default=-1, type=int)
parser.add_argument('--nb_samples', default=10000000, type=int, help='Max number of samples to train on')
parser.add_argument('--seed', default=123456, type=int, help='Random generator seed for train/test validation splitting')

args = parser.parse_args()

model_dir = args.model_dir  # каталог для файлов модели - при тренировке туда записываются, при векторизации - оттуда загружаются
tmp_dir = args.tmp_dir  # каталог для всяких сводок по процессу обучения
out_file = args.out_file  # в этот файл будет сохранены векторы слов в word2vec-совместимом формате
do_train = args.train  # тренировать ли модель с нуля
do_vectorize = args.vectorize  # векторизовать ли входной список слов
vec_size = args.dims  # размер вектора представления слова для тренировки модели
char_dims = args.char_dims  # если векторы символов будут меняться при тренировке, то явно надо задавать размерность векторов символов
batch_size = args.batch_size  # размер минибатчей существенно влияет на точность, поэтому разрешаем задавать его
min_batch_size = args.min_batch_size
tunable_char_embeddings = args.tunable_char_embeddings  # делать ли настраиваемые векторы символов (True) или 1-hot (False)
nb_samples = args.nb_samples  # макс. число слов, используемых для обучения
seed = args.seed

# архитектура модели:
# cnn - сверточный энкодер
# rnn - рекуррентный энкодер
# lstm+cnn - гибридная сетка с параллельными рекуррентными и сверточными слоями
# lstm(cnn) - сверточные слои и поверх них рекуррентные слои.
arch_type = args.arch_type

# настраиваем логирование в файл
logging_helpers.init_trainer_logging(os.path.join(tmp_dir, 'wordchar2vector.log'))

trainer = Wordchar2Vector_Trainer(arch_type,
                                  tunable_char_embeddings,
                                  char_dims,
                                  model_dir,
                                  vec_size,
                                  batch_size,
                                  min_batch_size,
                                  seed=seed)

if do_train:
    trainer.train(input_path, tmp_dir, nb_samples)

if do_vectorize:
    trainer.vectorize(input_path, out_file)

logging.info('Done.')
