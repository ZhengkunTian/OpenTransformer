"""
@Author: Zhengkun Tian
@Email: zhengkun.tian@outlook.com
@Date: 2020-04-23 15:14:28
@LastEditTime: 2020-04-23 15:16:49
@FilePath: \OpenTransformer\otrans\data.py
"""
import os
import torch
import random
import kaldiio as kio
import numpy as np
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator

PAD = 0
EOS = 1
BOS = 1
UNK = 2
MASK = 2
unk = '<UNK>'
compute_fbank = ta.compliance.kaldi.fbank


def load_vocab(vocab_file):
    # unit2idx = {'<S/E>': 0, '<PAD>': 1, '<UNK>': 2}
    unit2idx = {}
    with open(os.path.join(vocab_file), 'r', encoding='utf-8') as v:
        for line in v:
            unit, idx = line.strip().split()
            unit2idx[unit] = int(idx)
    return unit2idx


def normalization(feature):
    std, mean = torch.std_mean(feature, dim=0)
    return (feature - mean) / std


def apply_cmvn(mat, stats):
    mean = stats[0, :-1] / stats[0, -1]
    variance = stats[1, :-1] / stats[0, -1] - np.square(mean)
    return np.divide(np.subtract(mat, mean), np.sqrt(variance))


def spec_augment(
    mel_spectrogram,
    freq_mask_num=2,
    time_mask_num=2,
    freq_mask_rate=0.3,
    time_mask_rate=0.05,
    max_mask_time_len=100):

    tau = mel_spectrogram.shape[0]
    v = mel_spectrogram.shape[1]

    warped_mel_spectrogram = mel_spectrogram

    freq_masking_para = int(v * freq_mask_rate)
    time_masking_para = min(int(tau * time_mask_rate), max_mask_time_len)

    # Step 1 : Frequency masking
    if freq_mask_num > 0:
        for _ in range(freq_mask_num):
            f = np.random.uniform(low=0.0, high=freq_masking_para)
            f = int(f)
            f0 = random.randint(0, v-f)
            warped_mel_spectrogram[:, f0:f0+f] = 0

    # Step 2 : Time masking
    if time_mask_num > 0:
        for _ in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=time_masking_para)
            t = int(t)
            t0 = random.randint(0, tau-t)
            warped_mel_spectrogram[t0:t0+t, :] = 0

    return warped_mel_spectrogram

class AudioDataset(Dataset):
    def __init__(self, params, name='train', is_eval=False):

        self.params = params
        self.is_eval = is_eval

        self.unit2idx = load_vocab(params['vocab'])

        if params['from_kaldi']:
            self.from_kaldi = True
            print('Load Kaldi Features!')
        else:
            self.from_kaldi = False
            print('Extract Features ONLINE!')

        self.targets_dict = {}
        with open(os.path.join(params[name], params['text']), 'r', encoding='utf-8') as t:
            for line in t:
                parts = line.strip().split()
                utt_id = parts[0]
                label = []
                for c in parts[1:]:
                    label.append(self.unit2idx[c] if c in self.unit2idx else self.unit2idx[unk])
                self.targets_dict[utt_id] = label

        self.file_list = []
        with open(os.path.join(params[name], 'feats.scp' if self.from_kaldi else 'wav.scp'), 'r', encoding='utf-8') as fid:
            for line in fid:
                idx, path = line.strip().split()
                self.file_list.append([idx, path])

        assert len(self.file_list) == len(
            self.targets_dict), 'please keep feats.scp and %s have the same lines.' % params['text']

        self.lengths = len(self.file_list)

        if params['apply_cmvn']:

            assert os.path.isfile(os.path.join(params[name], 'utt2spk'))
            assert os.path.isfile(os.path.join(params[name], 'cmvn.scp'))

            self.utt2spk = {}
            with open(os.path.join(params[name], 'utt2spk'), 'r') as f:
                for line in f:
                    utt_id, spk_id = line.strip().split()
                    self.utt2spk[utt_id] = spk_id
                print('Load Speaker INFO')

            self.cmvns = {}
            with open(os.path.join(params[name], 'cmvn.scp'), 'r') as f:
                for line in f:
                    spk_id, path = line.strip().split()
                    self.cmvns[spk_id] = path
                print('Load CMVN Stats')

        self.apply_spec_augment = self.params['spec_augment'] if not self.is_eval else False
        if self.apply_spec_augment:
            print('Apply SpecAugment!')

    def __getitem__(self, index):
        utt_id, path = self.file_list[index]

        if self.from_kaldi:
            feature = kio.load_mat(path)
        else:
            wavform, sample_frequency = ta.load_wav(path)
            feature = compute_fbank(wavform, num_mel_bins=self.params['num_mel_bins'], sample_frequency=sample_frequency, dither=0.0)

        if self.params['apply_cmvn']:
            spk_id = self.utt2spk[utt_id]
            stats = kio.load_mat(self.cmvns[spk_id])
            feature = apply_cmvn(feature, stats)

        if self.params['normalization']:
            feature = normalization(feature)
            
        if self.apply_spec_augment:
            feature = spec_augment(feature)

        feature_length = feature.shape[0]
        targets = self.targets_dict[utt_id]
        targets_length = len(targets)

        return utt_id, feature, feature_length, targets, targets_length

    def __len__(self):
        return self.lengths

    def read_features(self, path):
        raise NotImplementedError

    def encode(self, seq):

        encoded_seq = []
        if self.encoding:
            for c in seq:
                if c in self.unit2idx:
                    encoded_seq.append(self.unit2idx[c])
                else:
                    encoded_seq.append(self.unit2idx['<UNK>'])
        else:
            encoded_seq = [int(i) for i in seq]

        return encoded_seq

    @property
    def idx2char(self):
        return {i: c for (c, i) in self.unit2idx.items()}

    @property
    def vocab_size(self):
        return len(self.unit2idx)

    @property
    def batch_size(self):
        return self.params['batch_size']


def audio_collate_fn(batch):

    utt_ids = [data[0] for data in batch]
    features_length = [data[2] for data in batch]
    targets_length = [data[4] for data in batch]
    max_feature_length = max(features_length)
    max_target_length = max(targets_length)

    padded_features = []
    padded_targets = []

    for _, feat, feat_len, target, target_len in batch:
        padded_features.append(np.pad(feat, ((
            0, max_feature_length-feat_len), (0, 0)), mode='constant', constant_values=0.0))
        padded_targets.append(
            [BOS] + target + [EOS] + [PAD] * (max_target_length - target_len))

    features = torch.FloatTensor(padded_features)
    features_length = torch.IntTensor(features_length)
    targets = torch.LongTensor(padded_targets)
    targets_length = torch.IntTensor(targets_length)

    inputs = {
        'inputs': features,
        'inputs_length': features_length,
        'targets': targets,
        'targets_length': targets_length
    }
    return utt_ids, inputs


class TextDataset(Dataset):
    def __init__(self, params, name='train', is_eval=False):

        self.params = params
        self.is_eval = is_eval
        self.unit2idx = load_vocab(params['vocab'])

        self.text_list = []
        with open(params[name], 'r', encoding='utf-8') as t:
            for line in t:
                parts = line.strip().split()
                utt_id = parts[0]
                label = []
                for c in parts[1:]:
                    label.append(self.unit2idx[c] if c in self.unit2idx else self.unit2idx['<UNK>'])
                self.text_list.append((utt_id, label))

        self.lengths = len(self.text_list)

    def __getitem__(self, index):
        idx, seq = self.text_list[index]
        return idx, seq, seq

    def __len__(self):
        return self.lengths

    @property
    def vocab_size(self):
        return len(self.unit2idx)

    @property
    def idx2unit(self):
        return {i: c for (c, i) in self.unit2idx.items()}


def text_collate_fn(batch):

    utt_ids = [data[0] for data in batch]
    src_length = [len(data[1]) + 1 for data in batch]
    tgt_length = [len(data[2]) + 1 for data in batch]

    max_src_length = max(src_length)
    max_tgt_length = max(tgt_length) 

    padded_src = []
    padded_tgt = []

    for _, src_seq, tgt_seq in batch:
        padded_src.append([BOS] + src_seq + [PAD] * (max_src_length - len(src_seq) - 1))
        padded_tgt.append(tgt_seq + [EOS] + [PAD] * (max_tgt_length - len(tgt_seq) - 1))

    src_seqs = torch.LongTensor(padded_src)
    tgt_seqs = torch.LongTensor(padded_tgt)

    inputs = {
        'inputs': src_seqs,
        'targets': tgt_seqs
    }
    return utt_ids, inputs


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class FeatureLoader(object):
    def __init__(self, params, name, shuffle=False, ngpu=1, mode='dp', is_eval=False):

        self.ngpu = ngpu
        self.shuffle = False if is_eval else shuffle

        self.dataset_type = params['data']['dataset_type']   # text, online, espnet
        self.batch_size = params['data']['batch_size']

        if self.dataset_type == 'text':
            self.dataset = TextDataset(params['data'], name, is_eval=is_eval)
            collate_fn = text_collate_fn
        else:
            self.dataset = AudioDataset(params['data'], name, is_eval=is_eval)
            collate_fn = audio_collate_fn

        if ngpu > 1 and mode == 'ddp':
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        else:
            self.sampler = None

        if ngpu >= 1:
            self.batch_size *= ngpu

        self.loader = DataLoaderX(
            self.dataset, batch_size=self.batch_size,
            shuffle=self.shuffle if self.sampler is None else False,
            num_workers=ngpu, pin_memory=True, sampler=self.sampler,
            collate_fn=collate_fn
        )

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
