# _*_ coding: utf-8 _*_

import os
from gensim.models.doc2vec import TaggedDocument


class SplitSentences(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def __iter__(self):
        for file_name in os.listdir(self.dir_name):
            with open(os.path.join(self.dir_name, file_name), encoding='utf-8') as f:
                for line in f.readlines():
                    yield line.split('\\s+')


class TaggedSentences(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def __iter__(self):
        index = 0
        for file_name in os.listdir(self.dir_name):
            with open(os.path.join(self.dir_name, file_name), encoding='utf-8') as f:
                for line in f.readlines():
                    index += 1
                    yield TaggedDocument(line.split('\\s+'), [index])


class TaggedDocuments(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def __iter__(self):
        index = 0
        for file_name in os.listdir(self.dir_name):
            with open(os.path.join(self.dir_name, file_name), encoding='utf-8') as f:
                tlist = []
                for line in f.readlines():
                    ltmp = line.split('\\s+')
                    for tmp in ltmp:
                        tlist.append(tmp)
                yield TaggedDocument(tlist, [index])
            index += 1
