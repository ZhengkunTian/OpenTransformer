# File   : kaldi.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import torch
import logging
import kaldiio as kio
import numpy as np
from otrans.data import load_vocab, UNK_TOKEN
from otrans.data.augment import spec_augment
from torch.utils.data import Dataset


class KaldiDataset(Dataset):
    def __init__(self, params, datadict, is_eval=False):

        self.params = params
        self.datadict = datadict
        self.is_eval = is_eval
        self.max_length = params['max_length'] if 'max_length' in params and not is_eval else 1000

        self.apply_spec_augment = True if params['spec_augment'] and not self.is_eval else False

        if self.apply_spec_augment:
            logging.info('Apply SpecAugment!')
            self.spec_augment_config = params['spec_augment_config']
            logging.info('Config: %s' % ' '.join([key+':'+str(value) for key, value in self.spec_augment_config.items()]))

        self.unit2idx = load_vocab(params['vocab'])

        self.target_dict = {}
        for text_file in self.datadict['text']:
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    utt_id = parts[0]
                    tokens = self.encode(parts[1:])
                    if len(tokens) > self.max_length: continue
                    self.target_dict[utt_id] = tokens

        self.feat_list = []
        for feat_file in self.datadict['feat']:
            with open(feat_file, 'r', encoding='utf-8') as f:
                for line in f:
                    utt_id, feat_path = line.strip().split()
                    if utt_id not in self.target_dict: continue
                    self.feat_list.append([utt_id, feat_path])

        if 'utt2spk' in self.datadict:
            self.apply_cmvn = True
            assert 'cmvn' in self.datadict
            self.utt2spk = {}
            for utt2spk in self.datadict['utt2spk']:
                with open(utt2spk, 'r') as f:
                    for line in f:
                        uttid, spkid = line.strip().split()
                        self.utt2spk[uttid] = spkid
            
            self.cmvn = {}
            for cmvn in self.datadict['cmvn']:
                with open(cmvn, 'r') as f:
                    for line in f:
                        spkid, path = line.strip().split()
                        self.cmvn[spkid] = path
            logging.info('Apply CMVN!')
        else:
            self.apply_cmvn = False

    def __getitem__(self, index):
        utt_id, feat_path = self.feat_list[index]

        feature = kio.load_mat(feat_path)

        if self.apply_cmvn:
            spkid = self.utt2spk[utt_id]
            stats = kio.load_mat(self.cmvn[spkid])
            mean = stats[0, :-1] / stats[0, -1]
            variance = stats[1, :-1] / stats[0, -1] - np.square(mean)
            feature =  np.divide(np.subtract(feature, mean), np.sqrt(variance))

        feature_length = feature.shape[0]
        feature = torch.FloatTensor(feature)

        if self.apply_spec_augment:
            feature = spec_augment(feature, **self.spec_augment_config)

        targets = self.target_dict[utt_id]
        targets_length = len(targets)

        return utt_id, feature, feature_length, targets, targets_length

    def __len__(self):
        return len(self.feat_list)

    def index_length_pair(self):  
        length_list = []    
        if 'feat-to-len' in self.datadict:
            logging.info('Load feat-to-len file for building buckets!')
            feat_to_len = {}

            for feat_to_len_file in self.datadict['feat-to-len']:
                with open(feat_to_len_file, 'r') as f:
                    for line in f:
                        uttid, lens = line.strip().split()
                        feat_to_len[uttid] = lens

            for index in range(len(self)):
                uttid, _ = self.feat_list[index]
                lens = int(feat_to_len[uttid])
                length_list.append((index, lens))
        else:
            logging.info('Compute the number of frames for building buckets!')
            for index in range(len(self)):
                _, feat_path = self.feat_list[index]
                feature = kio.load_mat(feat_path)
                length_list.append((index, feature.shape[0]))

        return length_list 

    def encode(self, seq):
        ids = []
        for s in seq:
            ids.append(self.unit2idx[s] if s in self.unit2idx else self.unit2idx[UNK_TOKEN])
        return ids

    @property
    def idx2unit(self):
        return {i: c for (c, i) in self.unit2idx.items()}
