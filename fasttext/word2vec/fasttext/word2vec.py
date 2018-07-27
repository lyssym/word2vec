# _*_ coding: utf-8 _*_

import multiprocessing
import fasttext
from fasttext import skipgram
from fasttext import cbow
from fasttext import load_model


class Word2Vector(object):
    '''fasttext对输入格式有要求，label标签使用  “__label__”+实际标签的形式'''
    def __init__(self, src_file, model_file, dim=300, ws=5, min_count=10, hs=0, sg=1,
                 learning_rate=0.025, epoch=5, word_ngrams=5, lr_update_rate=100):
        self.src_file = src_file
        self.model_file = model_file
        self.dim = dim
        self.ws = ws
        self.sg = sg
        self.min_count = min_count
        self.lr = learning_rate
        self.lr_update_rate = lr_update_rate
        self.epoch = epoch
        self.word_ngrams = word_ngrams
        self.thread = multiprocessing.cpu_count()

    def train(self):
        if self.sg:
            self.model = skipgram(self.src_file, self.model_file, dim=self.dim, ws=self.ws,
                                  min_count=self.min_count, lr=self.lr, lr_update_rate=self.lr_update_rate,
                                  epoch=self.epoch, word_ngrams=self.word_ngrams, thread=self.thread)
        else:
            self.model = cbow(self.src_file, self.model_file, dim=self.dim, ws=self.ws,
                              min_count=self.min_count, lr=self.lr, lr_update_rate=self.lr_update_rate,
                              epoch=self.epoch, word_ngrams=self.word_ngrams, thread=self.thread)

    def load_model(self, model_name):
        target = model_name + '.bin'
        self.model = load_model(target, encoding='utf-8')

    def show_similarity(self, word1, word2):
        return self.model.cosine_similarity(word1, word2)
