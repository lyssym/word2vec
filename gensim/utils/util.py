# _*_ coding: utf-8 _*_

import os


class SplitSentences(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def __iter__(self):
        for file_name in os.listdir(self.dir_name):
            with open(os.path.join(self.dir_name, file_name), encoding='utf-8') as f:
                for line in f.readlines():
                    yield line.split('\\s+')
