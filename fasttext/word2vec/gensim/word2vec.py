# _*_ coding: utf-8 _*_

import multiprocessing
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import PathLineSentences


class Word2Vector(object):
    def __init__(self, src_file, dst_file, size=300, window=5, min_count=10, hs=0, sg=0, learning_rate=0.025):
        self.src_file = src_file
        self.model_file = dst_file
        self.size = size
        self.window = window
        self.min_count = min_count
        self.hs = hs   # 1: 分层softmax, 0: 不使用分层softmax
        self.sg = sg   # 1: skip-gram,  0: CBOW
        self.alpha = learning_rate
        self.workers = multiprocessing.cpu_count()

    def train(self, sentences):
        self.model = FastText(sentences, size=self.size, window=self.window, min_count=self.min_count,
                         sg=self.sg, workers=self.workers)
        self.model.save(self.model_file)
        self.model.save_word2vec_format(self.model_file + '.bin', binary=True)

    def train_model(self):
        sentences = LineSentence(self.src_file)
        self.train(sentences)

    def online_train_model(self, sentences):  # 在线训练
        self.model.build_vocab(LineSentence(sentences))
        self.model.train(total_examples=self.model.corpus_count, epochs=self.model.iter)

    def online_train_model(self, file_name, isdir=True):  # 在线训练
        if isdir:
            sentences = PathLineSentences(self.src_file)
        else:
            sentences = LineSentence(self.src_file)
        self.online_train_model(sentences)

    def train_dir_model(self):
        sentences = PathLineSentences(self.src_file)
        self.train(sentences)

    def load_model(self, model_name):
        self.model = FastText.load(model_name)

    def show_similarity(self, word1, word2):
        return self.model.wv.similarity(word1, word2)

    def show_word_vector(self, word):
        return self.model.wv[word]
