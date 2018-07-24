# _*_ coding: utf-8 _*_

import multiprocessing



class Word2Vector(object):
    def __init__(self, src_file, model_file, dim=300, min_count=5, memory=4.0, epoch=100,
                 binary=2, m_max=10, window=15, verbose=2):
        self.src_file = src_file
        self.cooccurrent_file = 'cooccurrence.bin'
        self.vector_size = dim
        self.vocab_min_count = min_count
        self.memory = memory
        self.max_iter = epoch
        self.window_size = window
        self.verbose = verbose
        self.binary = binary
        self.m_max = m_max
        self.num_threads = multiprocessing.cpu_count()
