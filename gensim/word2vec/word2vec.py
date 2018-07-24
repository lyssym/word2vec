# _*_ coding: utf-8 _*_

import multiprocessing
from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import PathLineSentences
from gensim.models import word2vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from gensim.utils.util import SplitSentences


class Word2Vector(object):
    def __init__(self, src_file, dst_file, size=300, window=5, min_count=10, sg=0, learning_rate=0.025):
        self.src_file = src_file
        self.model_file = dst_file
        self.size = size
        self.window = window
        self.min_count = min_count
        self.sg = sg   # 1 : skip-gram, 0: CBOW
        self.alpha = learning_rate
        self.workers = multiprocessing.cpu_count()

    def train(self, sentences):
        model = Word2Vec(sentences, size=self.size, window=self.window, min_count=self.min_count,
                         sg=self.sg, workers=self.workers)
        model.save(self.model_file)
        model.save_word2vec_format(self.model_file + '.bin', binary=True)

    def train_model(self):
        sentences = LineSentence(self.src_file)
        self.train(sentences)

    def train_dir_model(self, custom=True):
        if custom:
            sentences = SplitSentences(self.src_file)
        else:
            sentences = PathLineSentences(self.src_file)
        self.train(sentences)

    def load_model(self, model_name):
        self.model = word2vec.Word2Vec.load(model_name)

    def load_google_model(self, model_name, binary=True):
        self.model = KeyedVectors.load_word2vec_format(datapath(model_name), binary=binary)

    def show_similarity(self, word1, word2):
        return self.model.wv.similarity(word1, word2)

    def show_similarity_by_word(self, word, topn=10):
        return self.model.wv.similar_by_word(word, topn)
