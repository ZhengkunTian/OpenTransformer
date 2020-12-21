# File   : text.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import torch
import logging
from otrans.data import *
from torch.utils.data import Dataset



class TextDataset(Dataset):
    def __init__(self, params, name='train', is_eval=False):

        self.params = params
        self.is_eval = is_eval
        self.src_unit2idx = load_vocab(params['src_vocab'])
        self.tgt_unit2idx = load_vocab(params['tgt_vocab'])
        self.reverse = params['reverse']

        if self.reverse:
            logging.info('Reverse the src and tgt sequence!')

        self.src_list = []
        self.tgt_dict = {}
        for src_file in params[name]['src']:
            with open(src_file, 'r', encoding='utf-8') as t:
                for line in t:
                    parts = line.strip().split()
                    utt_id = parts[0]
                    label = []
                    for c in parts[1:]:
                        label.append(self.src_unit2idx[c] if c in self.src_unit2idx else self.src_unit2idx[UNK_TOKEN])
                    self.src_list.append((utt_id, label))

        for tgt_file in params[name]['tgt']:
            with open(tgt_file, 'r', encoding='utf-8') as t:
                for line in t:
                    parts = line.strip().split()
                    utt_id = parts[0]
                    label = []
                    for c in parts[1:]:
                        label.append(self.tgt_unit2idx[c] if c in self.tgt_unit2idx else self.tgt_unit2idx[UNK_TOKEN])
                    self.tgt_dict[utt_id] = label

        assert len(self.src_list) == len(self.tgt_dict)

        self.lengths = len(self.src_list)

    def __getitem__(self, index):
        idx, src_seq = self.src_list[index]
        tgt_seq = self.tgt_dict[idx]

        if self.reverse:
            src_seq.reverse()
            tgt_seq.reverse()

        return idx, src_seq, tgt_seq

    def __len__(self):
        return self.lengths

    @property
    def src_vocab_size(self):
        return len(self.src_unit2idx)

    @property
    def tgt_vocab_size(self):
        return len(self.tgt_unit2idx)
    
    @property
    def src_idx2unit(self):
        return {i: c for (c, i) in self.src_unit2idx.items()}

    @property
    def tgt_idx2unit(self):
        return {i: c for (c, i) in self.tgt_unit2idx.items()}

