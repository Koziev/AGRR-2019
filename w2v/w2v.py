# -*- coding: utf-8 -*-
"""
Генерация word2vector моделей слов с использованием gensim.

Используется готовый корпус, в котором каждое слово отделено пробелами, и каждое
предложение находится на отдельной строке.
"""

from __future__ import print_function

import logging
import itertools
from gensim.models import word2vec


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Тренируемся на этом подготовленном текстовом корпусе
corpus_path = '/media/inkoziev/corpora/Corpus/word2vector/ru/w2v.ru.corpus.txt'

# основные параметры обучения модели
SIZE = 100
WINDOW = 5
CBOW = 1
MIN_COUNT = 2
NB_ITERS = 1
ADD_WALLS = True  # добавлять спецтокены <s> и </s> в начале и конце каждого предложения


# Если нужно добавить спецтокены <s> и </s> вокруг каждого предложения
class SentencesWithWalls(object):
    def __init__(self, fname):
        self.fname = fname
        self.rdr = None
        self.total_lines = 0

    def __iter__(self):
        self.total_lines = 0

        if self.rdr is not None:
            self.rdr.close()

        self.rdr = open(self.fname, 'r')
        return self

    def next(self):
        line = self.rdr.readline()
        if not line:
            raise StopIteration

        self.total_lines += 1
        return list(itertools.chain(['<s>'], line.decode('utf-8').strip().split(), ['</s>']))


# в названии файла модели сохраним ключевые параметры
filename = 'w2v.CBOW=' + str(CBOW)+'_WIN=' + str(WINDOW) + '_DIM='+str(SIZE)

# в отдельный текстовый файл выведем все параметры модели
with open( filename + '.info', 'w+') as info_file:
    print('corpus_path=', corpus_path, file=info_file)
    print('SIZE=', SIZE, file=info_file)
    print('WINDOW=', WINDOW, file=info_file)
    print('CBOW=', CBOW, file=info_file)
    print('MIN_COUNT=', MIN_COUNT, file=info_file)
    print('NB_ITERS=', NB_ITERS, file=info_file)

if ADD_WALLS:
    sentences = SentencesWithWalls(corpus_path)
else:
    #sentences = word2vec.Text8Corpus(corpus_path)
    sentences = word2vec.LineSentence(corpus_path)

# начинаем обучение w2v
model = word2vec.Word2Vec(sentences,
                          size=SIZE,
                          window=WINDOW,
                          cbow_mean=CBOW,
                          min_count=MIN_COUNT,
                          workers=6,
                          sorted_vocab=1,
                          iter=NB_ITERS)

model.init_sims(replace=True)

# сохраняем готовую w2v модель
model.wv.save_word2vec_format(filename + '.bin', binary=True)
#model.wv.save_word2vec_format(filename + '.txt', binary=False)
