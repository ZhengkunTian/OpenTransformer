# File   : loader.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import torch
import torch.nn.functional as F
from otrans.data.bucket import BySequenceLengthSampler
from otrans.data.audio import AudioDataset
from otrans.data.text import TextDataset
from otrans.data.espnet import ESPNetDataset
from otrans.data.kaldi import KaldiDataset
from otrans.data.generator import DataLoaderX
from otrans.data import EOS, PAD, BOS


Dataset = {
    'text': TextDataset,
    'online': AudioDataset,
    'espnet': ESPNetDataset,
    'kaldi': KaldiDataset
}


def text_collate_fn(batch):

    utt_ids = [data[0] for data in batch]
    src_length = [len(data[1]) for data in batch]
    tgt_length = [len(data[2]) for data in batch]

    max_src_length = max(src_length)
    max_tgt_length = max(tgt_length) 

    padded_src = []
    padded_tgt = []
    padded_source_mask = []
    padded_target_mask = []


    for _, src_seq, tgt_seq in batch:
        padded_source_len = max_src_length - len(src_seq)
        padded_src.append([BOS] + src_seq + [PAD] * padded_source_len)
        padded_source_mask.append([1] * (len(src_seq) + 1) + [0] * padded_source_len)

        padded_target_len = max_tgt_length - len(tgt_seq)
        padded_tgt.append(tgt_seq + [EOS] + [PAD] * padded_target_len)
        padded_target_mask.append([1] * (len(tgt_seq) + 1) + [0] * padded_target_len)


    src_seqs = torch.LongTensor(padded_src)
    src_mask = torch.IntTensor(padded_source_mask) > 0
    tgt_seqs = torch.LongTensor(padded_tgt)
    tgt_mask = torch.IntTensor(padded_target_mask) > 0

    inputs = {
        'inputs': src_seqs,
        'mask': src_mask,
    }

    targets = {
        'targets': tgt_seqs,
        'mask': tgt_mask
    }
    return utt_ids, inputs, targets


def collate_fn_with_eos_bos(batch):

    utt_ids = [data[0] for data in batch]
    features_length = [data[2] for data in batch]
    targets_length = [data[4] for data in batch]
    max_feature_length = max(features_length)
    max_target_length = max(targets_length)

    padded_features = []
    padded_targets = []
    padded_feature_mask = []
    padded_target_mask = []

    for _, feat, feat_len, target, target_len in batch:
        padding_feature_len = max_feature_length - feat_len
        padded_features.append(F.pad(feat, pad=(0, 0, 0, padding_feature_len), value=0.0).unsqueeze(0))
        padded_feature_mask.append([1] * feat_len + [0] * padding_feature_len)

        padding_target_len = max_target_length - target_len
        padded_targets.append([BOS] + target + [EOS] + [PAD] * padding_target_len)
        padded_target_mask.append([1] * (target_len + 2) + [0] * padding_target_len)

    features = torch.cat(padded_features, dim=0)
    features_length = torch.IntTensor(features_length)
    feature_mask = torch.IntTensor(padded_feature_mask) > 0

    targets = torch.LongTensor(padded_targets)
    targets_length = torch.IntTensor(targets_length).add(1)
    targets_mask = torch.IntTensor(padded_target_mask) > 0

    inputs = {
        'inputs': features,
        'inputs_length': features_length,
        'mask': feature_mask
    }

    targets = {
        'targets': targets,
        'targets_length': targets_length, # include eos
        'mask': targets_mask
    }

    return utt_ids, inputs, targets



class FeatureLoader(object):
    def __init__(self, params, name, ngpu=1, mode='dp', is_eval=False):

        self.ngpu = ngpu
        self.shuffle = False if is_eval else True
        self.num_workers = params['data']['num_workers'] if 'num_workers' in params['data'] else ngpu

        self.dataset_type = params['data']['dataset_type']   # text, online, espnet
        datadict = params['data'][name]
        self.dataset = Dataset[self.dataset_type](params['data'], datadict, is_eval=is_eval)

        if ngpu > 1 and mode == 'ddp':
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        else:
            self.sampler = None

        if 'bucket' in params['data'] and not is_eval:
            self.apply_bucket = True

            if self.ngpu > 1:
                params['data']['bucket']['max_frames_one_batch'] *= self.ngpu

            self.bucket_sampler = BySequenceLengthSampler(
                self.dataset, short_first=params['data']['short_first'] if 'short_first' in params['data'] else False,
                **params['data']['bucket']
            )

            self.batch_size = 1
            self.shuffle = False
        else:
            self.apply_bucket = False

            # if is_eval and ('eval' in params and 'batch_size' in params['eval']):
            #     self.batch_size = params['data']['batch_size']
            # else:
            self.batch_size = params['data']['batch_size']

            if ngpu >= 1:
                self.batch_size *= ngpu

        self.loader = DataLoaderX(
            self.dataset, batch_size=self.batch_size,
            shuffle=self.shuffle, sampler=self.sampler,
            num_workers=self.num_workers, pin_memory=True,
            batch_sampler=self.bucket_sampler if self.apply_bucket else None ,
            collate_fn=collate_fn_with_eos_bos if self.dataset_type != 'text' else text_collate_fn
        )

    def set_epoch(self, epoch):

        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        
        if self.apply_bucket:
            self.bucket_sampler.shuffle_batch_in_bucket()



class FeatureLoaderDIY(object):
    def __init__(self, params, datadict, ngpu=1, mode='dp', is_eval=False):

        self.ngpu = ngpu
        self.shuffle = False if is_eval else True
        self.num_workers = params['data']['num_workers'] if 'num_workers' in params else ngpu

        self.dataset_type = params['data']['dataset_type']   # text, online, espnet
        # datadict = params['data'][name]
        self.dataset = Dataset[self.dataset_type](params['data'], datadict, is_eval=is_eval)

        if ngpu > 1 and mode == 'ddp':
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        else:
            self.sampler = None

        if 'bucket' in params['data'] and not is_eval:
            self.apply_bucket = True

            if self.ngpu > 1:
                params['data']['bucket']['max_frames_one_batch'] *= self.ngpu

            self.bucket_sampler = BySequenceLengthSampler(
                self.dataset, short_first=params['data']['short_first'] if 'short_first' in params['data'] else False,
                **params['data']['bucket']
            )

            self.batch_size = 1
            self.shuffle = False
        else:
            self.apply_bucket = False

            # if is_eval and ('eval' in params and 'batch_size' in params['eval']):
            #     self.batch_size = params['data']['batch_size']
            # else:
            self.batch_size = params['data']['batch_size']

            if ngpu >= 1:
                self.batch_size *= ngpu

        self.loader = DataLoaderX(
            self.dataset, batch_size=self.batch_size,
            shuffle=self.shuffle, sampler=self.sampler,
            num_workers=self.num_workers, pin_memory=True,
            batch_sampler=self.bucket_sampler if self.apply_bucket else None ,
            collate_fn=collate_fn_with_eos_bos if self.dataset_type != 'text' else text_collate_fn
        )

    def set_epoch(self, epoch):

        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        
        if self.apply_bucket:
            self.bucket_sampler.shuffle_batch_in_bucket()