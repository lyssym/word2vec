# _*_ coding: utf-8 _*_

import multiprocessing
from gensim.models.doc2vec import Doc2Vec


class Doc2Vector(object):
    def __init__(self, documents, dim=200, dm=1, window=15, min_count=5, workers=8, epochs=40,
                 alpha=0.025, hs=1, dm_mean=0, dm_concat=1):
        self.size = dim
        self.dm = dm          # 1: distributed memory, 0: distributed bag of words
        self.window = window  # 窗口大小
        self.min_count = min_count
        self.workers = multiprocessing.cpu_count()
        self.iter = epochs
        self.alhpa = alpha
        self.hs = hs  # 分层softmax
        self.dm_mean = dm_mean
        self.dm_concat = dm_concat
        self.documents = documents

    def train(self, target_name):
        model = Doc2Vec(vector_size=self.size, dm=self.dm, alpha=self.alhpa, hs=self.hs,
                        dm_mean=self.dm_mean, dm_concat=self.dm_concat,
                        workers=self.workers, epochs=self.iter,
                        window=self.window, min_count=self.min_count)
        model.build_vocab(self.documents)
        model.train(self.documents, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(target_name)

    def load_model(self, model_name):
        self.model = Doc2Vec.load(model_name)

    def infer(self, tokens):
        return self.model.infer_vector(tokens)

