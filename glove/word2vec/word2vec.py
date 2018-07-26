# _*_ coding: utf-8 _*_

import multiprocessing

from glove.utils.util import run_command
from glove.utils.util import COOCCUR, VOCAB_COUNT, SHUFFLE, GLOVE
from glove.utils.util import COOCCURRENCE_FILE, COOCCURRENCE_SHUF_FILE, VOCAB_FILE
from glove.utils.util import GLOVE_BUILD


class Word2Vector(object):
    def __init__(self, src_file, model_file, dim=50, min_count=5, memory=8.0, epoch=15,
                 binary=2, x_max=10, alpha=0.025, window=15, verbose=2):
        self.src_file = src_file
        self.model_file = model_file
        self.vector_size = dim
        self.vocab_min_count = min_count
        self.memory = memory
        self.max_iter = epoch
        self.window_size = window
        self.verbose = verbose
        self.binary = binary
        self.x_max = x_max
        self.alpha = alpha
        self.num_threads = multiprocessing.cpu_count()*2

    def train(self):
        cmd = 'vocab_count -min-count %s -verbose %s < %s > %s' %(self.vocab_min_count, self.verbose,
                                                                  self.src_file, VOCAB_FILE)
        run_command(cmd)
        cmd = 'cooccur -memory %s -vocab-file %s -verbose %s -window-size %s < %s > %s ' %(self.memory, VOCAB_FILE,
                                                                                           self.verbose, self.window_size,
                                                                                           self.src_file, COOCCURRENCE_FILE)
        run_command(cmd)
        cmd = 'shuffle -memory %s -verbose %s < %s > %s' %(self.memory, self.verbose,
                                                           COOCCURRENCE_FILE, COOCCURRENCE_SHUF_FILE)
        run_command(cmd)
        cmd = 'glove -save-file %s -threads %s -input-file %s -x-max %s -iter %s -vector-size %s -alpha %s ' \
              '-binary %s -vocab-file %s -verbose %s' %(self.model_file, self.num_threads, COOCCURRENCE_SHUF_FILE,
                                                        self.x_max, self.max_iter, self.vector_size, self.alpha,
                                                        self.binary, VOCAB_FILE, self.verbose)
        run_command(cmd)


if __name__ == '__main__':
    src = 'text8'
    model = 'model'
    wv = Word2Vector(src, model, alpha=0.050, x_max=85)
    wv.train()

