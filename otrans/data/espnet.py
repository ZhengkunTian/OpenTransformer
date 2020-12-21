# File   : espnet.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import json
import torch
import logging
import kaldiio as kio
from otrans.data import load_vocab, UNK_TOKEN
from otrans.data.augment import spec_augment
from torch.utils.data import Dataset


class ESPNetDataset(Dataset):
    def __init__(self, params, datadict, is_eval=False):

        self.params = params
        self.datadict = datadict
        self.is_eval = is_eval
        self.apply_spec_augment = True if params['spec_augment'] and not self.is_eval else False

        if self.apply_spec_augment:
            logging.info('Apply SpecAugment!')
            self.spec_augment_config = params['spec_augment_config']
            logging.info('Config: %s' % ' '.join([key+':'+str(value) for key, value in self.spec_augment_config.items()]))

        self.unit2idx = load_vocab(params['vocab'])

        with open(self.datadict['json'], 'r') as f:
            self.utts = [(k, v) for k, v in json.load(f)['utts'].items()]

    def __getitem__(self, index):
        utt_id, infos = self.utts[index]

        path = infos['input'][0]['feat']
        feature = kio.load_mat(path)
        feature_length = feature.shape[0]
        feature = torch.FloatTensor(feature)
        if self.apply_spec_augment:
            feature = spec_augment(feature, **self.spec_augment_config)

        targets = infos['output'][0]['token']
        targets = self.encode(targets)
        targets_length = len(targets)

        return utt_id, feature, feature_length, targets, targets_length

    def __len__(self):
        return len(self.utts)

    def index_length_pair(self):
        length_list = []
        for index in range(len(self)):
            _, infos = self.utts[index]
            length = int(infos['input'][0]['shape'][0])
            length_list.append((index, length))
        return length_list

    def encode(self, seq):
        ids = []
        for s in seq.split():
            ids.append(self.unit2idx[s] if s in self.unit2idx else self.unit2idx[UNK_TOKEN])
        return ids

    @property
    def idx2unit(self):
        return {i: c for (c, i) in self.unit2idx.items()}
