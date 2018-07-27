# _*_ coding: utf-8 _*_

from gensim.models.wrappers.wordrank import Wordrank


class Word2Vector(object):
    '''基于Glove共生矩阵及排序训练词向量'''
    def __init__(self, binary_path, model_name, src_file, size=200, window=10, min_count=5, lr=0.025,
                 iter=10, epsilon=0.75, alpha=100, beta=99):
        self.binary_path = binary_path
        self.model_name = model_name
        self.src_file = src_file
        self.size = size
        self.window = window
        self.min_count = min_count
        self.lr = lr
        self.iter = iter
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta

    def train(self):
        self.model = Wordrank.train(self.binary_path, corpus_file=self.src_file, out_name=self.model_name,
                                    window=self.window, size=self.size, min_count=self.min_count,
                                    lrate=self.lr, iter=self.iter, epsilon=self.epsilon, alpha=self.alpha,
                                    beta=self.beta)

    def load_model(self, model_name):
        self.model = Wordrank.load_wordrank_model(model_file=model_name)

    def show_similarity(self, word1, word2):
        return self.model.distance(word1, word2)


if __name__ == '__main__':
    binary = '/home/liuyong/Documents/utils/wordrank'
    # model_name = 'target'
    # src = 'text8'
    # wv = Word2Vector(binary, model_name, src)
    # print('training')

    model = Wordrank.train(binary, corpus_file='text8', out_name='wr_model', iter=10, memory=8.0, np=4)
