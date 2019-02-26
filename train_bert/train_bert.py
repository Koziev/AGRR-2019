# -*- coding: utf-8 -*-
"""
Тренировка BERT на небольшом корпусе для последующего fine tuning'а в классификаторе предложений.
"""

import numpy as np
import keras
import os
import codecs
import collections
import pickle
import json
import itertools
import pandas as pd
import glob
import io
import csv
import sentencepiece as spm

from keras import Model
from keras_bert import get_base_dict, get_model, gen_batch_inputs
from sklearn.model_selection import train_test_split

from rutokenizer import Tokenizer
from rutokenizer import Segmenter

batch_size = 400
max_seq_len = 100  # макс. длина предложений, кол-во sentencepiece элементов, т.е. примерно в 3 раза больше, чем слов.
nb_epochs = 200
spm_items = 8000  # при обучении sentencepiece ограничиваем словарь модели таким количеством элементов


tmp_folder = '../tmp'
corpus_path = '/media/inkoziev/corpora/Corpus/Raw/ru/text_blocks.txt'


def split_str(s):
    #return tokenizer.tokenize(phrase1)
    return sp.EncodeAsPieces(s)
    #return list(itertools.chain(*(word2pieces(word) for word in s.split())))



segmenter = Segmenter()
tokenizer = Tokenizer()
tokenizer.load()


# --------------- SENTENCEPIECE ----------------------

# Готовим корпус для обучения SentencePiece
sentencepiece_corpus = os.path.join(tmp_folder, 'sentencepiece_corpus.txt')

nb_from_corpus = 0
max_nb_samples = 10000000  # макс. кол-во предложений для обучения SentencePiece
with io.open(sentencepiece_corpus, 'w', encoding='utf-8') as wrt:
    for file in glob.glob(os.path.join('../data', '*.csv')):
        print(u'Loading samples from {}...'.format(file))
        df = pd.read_csv(file, encoding='utf-8', delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for irow, row in df.iterrows():
            text = row['text']
            phrases = segmenter.split(text)
            for phrase in phrases:
                wrt.write(u'{}\n'.format(phrase))
                nb_from_corpus += 1

            if nb_from_corpus >= max_nb_samples:
                break

    print(u'Loading samples from {}'.format(corpus_path))
    with codecs.open(corpus_path, 'r', 'utf-8') as rdr:
        for line in rdr:
            line = line.strip()
            phrases = segmenter.split(line)
            for phrase in phrases:
                wrt.write(u'{}\n'.format(phrase))
                nb_from_corpus += 1

            if nb_from_corpus >= max_nb_samples:
                break

spm_name = 'sentencepiece4bert_{}'.format(spm_items)

if not os.path.exists(os.path.join(tmp_folder, spm_name + '.vocab')):
    spm.SentencePieceTrainer.Train(
        '--input={} --model_prefix={} --vocab_size={} --shuffle_input_sentence=true --character_coverage=1.0 --model_type=unigram'.format(
            sentencepiece_corpus, spm_name, spm_items))
    os.rename(spm_name + '.vocab', os.path.join(tmp_folder, spm_name + '.vocab'))
    os.rename(spm_name + '.model', os.path.join(tmp_folder, spm_name + '.model'))

sp = spm.SentencePieceProcessor()
rc = sp.Load(os.path.join(tmp_folder, spm_name + '.model'))
print('SentencePiece model loaded with status={}'.format(rc))


# Готовим корпус для обучения BERT
print('Building corpus for BERT...')
max_nb_samples = 100000  # макс. кол-во пар предложений для обучения и валидации BERT
sentence_pairs = []
nb_tokens = 0
nb_from_corpus = 0
all_words = collections.Counter()

for file in glob.glob(os.path.join('../data', '*.csv')):
    print(u'Loading samples from {}...'.format(file))
    df = pd.read_csv(file, encoding='utf-8', delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for irow, row in df.iterrows():
        text = row['text']
        phrases = segmenter.split(text)
        if len(phrases) > 1:
            for phrase1, phrase2 in zip(phrases, phrases[1:]):
                words1 = split_str(phrase1)[:max_seq_len - 4]
                words2 = split_str(phrase2)[:max_seq_len - 4]
                if len(words1) > 0 and len(words2) > 0:
                    nb_tokens += len(words1)
                    nb_from_corpus += 1
                    sentence_pairs.append((words1, words2))
                    all_words.update(words1+words2)

            if nb_from_corpus >= max_nb_samples:
                break

print(u'Loading samples from {}'.format(corpus_path))
with codecs.open(corpus_path, 'r', 'utf-8') as rdr:
    for line in rdr:
        line = line.strip()
        phrases = segmenter.split(line)
        for phrase1, phrase2 in zip(phrases, phrases[1:]):
            words1 = split_str(phrase1)[:max_seq_len - 4]
            words2 = split_str(phrase2)[:max_seq_len - 4]
            if len(words1) > 0 and len(words2) > 0:
                nb_tokens += len(words1)
                nb_from_corpus += 1
                sentence_pairs.append((words1, words2))
                all_words.update(words1+words2)

        if nb_from_corpus >= max_nb_samples:
            break

print('{} tokens in corpus'.format(nb_tokens))
print('vocabulary size={}'.format(len(all_words)))
print('{} samples'.format(len(sentence_pairs)))

with codecs.open('../tmp/vocab.csv', 'w', 'utf-8') as wrt:
    for word, freq in all_words.most_common():
        wrt.write(u'{}\t{}\n'.format(word, freq))

# Build token dictionary
token_dict = get_base_dict()  # A dict that contains some special tokens
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())  # Used for selecting a random word

tokens_path = '../tmp/token_dict.pickle'
with open(tokens_path, 'w') as f:
    pickle.dump(token_dict, f)

weights_path = '../tmp/bert.weights'

# Параметры BERT модели сохраним в файле, чтобы потом воссоздать архитектуру
bert_config = {'token_num': len(token_dict),
               'head_num': 6,  # было 4
               'transformer_num': 4,
               'embed_dim': 36,
               'feed_forward_dim': 150,  # было 100
               'seq_len': max_seq_len,
               'pos_num': max_seq_len,
               'dropout_rate': 0.05,
               }

with open('../tmp/bert.config', 'w') as f:
    json.dump(bert_config, f)

# Build & train the model
model = get_model(**bert_config)
model.summary()

#for layer in model.layers:
#    print('{}: {} --> {}'.format(layer.name, layer.input_shape, layer.output_shape))


def my_generator(samples, batch_size):
    while True:
        start_index = 0
        while (start_index + batch_size) < len(samples):
            if False:
                print(u'DEBUG\nstart_index={}\nphrase1 len={} words={}\nphrase2 len={} words={}\n'.format(start_index,
                                                                                                          len(samples[start_index][0]),
                                                                                                          u' '.join(samples[start_index][0]),
                                                                                                          len(samples[start_index][1]),
                                                                                                          u' '.join(samples[start_index][1])))

            yield gen_batch_inputs(samples[start_index: start_index + batch_size],
                                   token_dict,
                                   token_list,
                                   seq_len=max_seq_len,
                                   mask_rate=0.3,
                                   swap_sentence_rate=1.0)
            start_index += batch_size



SEED = 123456
TEST_SHARE = 0.2
samples_train, samples_val = train_test_split(sentence_pairs, test_size=TEST_SHARE, random_state=SEED)

model_checkpoint = keras.callbacks.ModelCheckpoint(weights_path,
                                                   monitor='val_loss',
                                                   verbose=1,
                                                   save_best_only=True,
                                                   mode='auto')

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

print('Start training on {} samples'.format(len(samples_train)))
hist = model.fit_generator(generator=my_generator(samples_train, batch_size),
                           steps_per_epoch=len(samples_train) // batch_size,
                           epochs=nb_epochs,
                           validation_data=my_generator(samples_val, batch_size),
                           validation_steps=len(samples_val) // batch_size,
                           callbacks=[model_checkpoint, early_stopping],
                           verbose=2)
model.load_weights(weights_path)

with open('../tmp/bert.learning_curve.csv', 'w') as f:
    for epoch, (l, vl) in enumerate(zip(hist.history['loss'], hist.history['val_loss'])):
        f.write('{}\t{}\t{}\n'.format(epoch+1, l, vl))


# `output_layer` is the last feature extraction layer (the last transformer)
# The input layers and output layer will be returned if `training` is `False`
inputs, output_layer = get_model(training=False, **bert_config)

model2 = Model(inputs=inputs, outputs=output_layer)
model2.summary()

#print('output_layer.output_shape={}'.format(output_layer.output_shape))

print('Copying the weights...')
for layer2 in model2.layers:
    layer2.set_weights(model.get_layer(layer2.name).get_weights())

#with open('./tmp/my_train_model2.arch', 'w') as f:
#    f.write(model.to_json())
#model2.save('../tmp/bert.model')
model2.save_weights(weights_path)

exit(0)

# Далее - отладочный код для проверки персистентности модели

input_sent = u'два маршрутизатора'
tokens = split_str(input_sent)

token_input = np.asarray([[token_dict[token] for token in tokens] + [0] * (max_seq_len - len(tokens))])
seg_input = np.asarray([[0] * len(tokens) + [0] * (max_seq_len - len(tokens))])

# Сначала посмотрим, что выдаст модель до выгрузки-загрузки
y_pred = model2.predict([token_input, seg_input])[0]
y_pred = y_pred.flatten()
print(y_pred)


# ---- тестируем загрузку и использование модели
with open('../tmp/bert.config', 'r') as f:
    bert_config = json.load(f)

inputs, output_layer = get_model(training=False, **bert_config)
model2 = Model(inputs=inputs, outputs=output_layer)
model2.load_weights(weights_path)

with open('../tmp/token_dict.pickle', 'r') as f:
    token_dict = pickle.load(f)

token_input = np.asarray([[token_dict[token] for token in tokens] + [0] * (max_seq_len - len(tokens))])
seg_input = np.asarray([[0] * len(tokens) + [0] * (max_seq_len - len(tokens))])

y_pred = model2.predict([token_input, seg_input])[0]
y_pred = y_pred.flatten()
print(y_pred)

