# _*_ coding: utf-8 _*_

import os
import sys

GLOVE_BUILD = 'build'
VOCAB_COUNT = os.path.join(GLOVE_BUILD, 'vocab_count')
COOCCUR = os.path.join(GLOVE_BUILD, 'cooccur')
SHUFFLE = os.path.join(GLOVE_BUILD, 'shuffle')
GLOVE = os.path.join(GLOVE_BUILD, 'glove')

COOCCUR


def run_command(cmd):
    print(cmd)
    if os.system(cmd) != 0:
        print('[ERROR] in running command "%s"' %cmd)
        sys.exit(-1)