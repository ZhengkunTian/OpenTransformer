# File   : __init__.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import os

BLK = 0
PAD = 0
BOS = 1
EOS = 1
UNK = 2
MASK = 3

BOS_TOKEN = '<S/E>'
EOS_TOKEN = '<S/E>'
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SPACE_TOKEN = '<SPACE>'
MASK_TOKEN = '<MASK>'

def load_vocab(vocab_file):
    unit2idx = {}
    with open(os.path.join(vocab_file), 'r', encoding='utf-8') as v:
        for line in v:
            unit, idx = line.strip().split()
            unit2idx[unit] = int(idx)
    return unit2idx


def load_idx2unit_map(vocab_file):

    unit2idx = load_vocab(vocab_file)
    idx2unit = {v: k for (k, v) in unit2idx.items()}

    return idx2unit
