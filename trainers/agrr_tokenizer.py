# -*- coding: utf-8 -*-

import io
import rutokenizer


class TokenizerAGRR(rutokenizer.Tokenizer):
    def __init__(self):
        super(TokenizerAGRR, self).__init__()
        self.known_words = None
        self.lat2cyr = {u'e': u'е', u'y': u'у', u'c': u'с', u'o': u'о', u'a': u'а', u'p': u'р', u't': u'т', u'x': u'х'}
        self.lat2cyr_keys = set(self.lat2cyr.keys())

    def load(self):
        super(TokenizerAGRR, self).load()
        self.known_words = set()
        with io.open('../data/wordforms.txt', 'r', encoding='utf-8') as rdr:
            for line in rdr:
                word = line.strip().lower().replace(' - ', '-')
                self.known_words.add(word)

    def normalize(self, t):
        t2 = t.lower().replace(u'ё', u'е')

        if u'‑' in t2:  # по‑немецки
            t3 = t2.replace(u'‑', u'-')
            if t3 in self.known_words:
                t2 = t3

        if u'­' in t2:
            t3 = t2.replace(u'­', u'')
            if t3 in self.known_words:
                t2 = t3

        if len(t2) > 2 and (set(t2) & self.lat2cyr_keys):
            t4 = u''
            for c in t2:
                if c in self.lat2cyr_keys:
                    t4 += self.lat2cyr[c]
                else:
                    t4 += c
            if t4 in self.known_words:
                t2 = t4

        return t2

    def tokenize2(self, phrase):
        raw_tokens = super(TokenizerAGRR, self).tokenize2(phrase)
        return [(self.normalize(t[0]), t[1], t[2]) for t in raw_tokens]
