# _*_ coding: utf-8 _*_

import os
import sys

VOCAB_FILE = 'vocab.txt'
COOCCURRENCE_FILE = 'cooccurrentce.bin'
COOCCURRENCE_SHUF_FILE = 'cooccurrentce.shuf.bin'


def run_command(cmd):
    print(cmd)
    if os.system(cmd) != 0:
        print('[ERROR] in running command "%s"' %cmd)
        sys.exit(-1)
