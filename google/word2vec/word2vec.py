# _*_ coding: utf-8 _*_

from word2vec import word2vec
from word2vec import load


##################################################################################
##########################  google word2vec python 调用接口 #######################
##########################  支持cbow 或者 skip-gram方式       ######################

class Word2Vector(object):
    def __init__(self, file_name, target_name, window=5, hs=1, learning_rate=0.025,
                 size=300, verbose=True):
        self.src_file = file_name
        self.model_file = target_name
        self.window = window
        self.hs = hs
        self.alpha = learning_rate
        self.size = size
        self.verbose = verbose

    def train_model(self):
        word2vec(self.src_file, self.model_file, window=self.window, hs=self.hs,
                 alpha=self.alpha, size=self.size, verbose=self.verbose)

    def load_model(self, model_name):
        self.model = load(model_name)

    def show_similarity(self, word):
        indexes = self.model.cosine(word)
        ret = []
        for index in indexes[0]:
            ret.add(self.model.vocab[index])
