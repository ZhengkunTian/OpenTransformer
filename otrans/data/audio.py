# File   : audio.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import torch
import random
import logging
import numpy as np
import torchaudio as ta
import scipy.io.wavfile as siw
import python_speech_features as psf
from otrans.data.augment import spec_augment
from torch.utils.data import Dataset
from otrans.data import load_vocab, UNK_TOKEN, EOS, BOS, PAD

logger = logging.getLogger(__name__)

# def normalization(feature):
#     std, mean = torch.std_mean(feature, dim=0)
#     return (feature - mean) / std

def normalization(feature):
    std, mean = torch.std_mean(feature)
    return (feature - mean) / std


class AudioDataset(Dataset):
    def __init__(self, params, datadict, is_eval=False):

        self.params = params
        self.datadict= datadict
        self.is_eval = is_eval
        self.apply_spec_augment = params['spec_augment'] if not self.is_eval else False

        logger.info('[Online-Reader] Read the feature extracted online!')

        self.normalization = params['normalization']
        self.feature_extractor = params['feature_extractor'] if 'feature_extractor' in params else 'torchaudio'
        assert self.feature_extractor in ['torchaudio', 'python_speech_feature', 'ta', 'psf']
        logger.info('Utilize %s to extract feature from wav.' % self.feature_extractor)
        if self.normalization:
            logger.info('Apply Feature Normalization!')
            if 'global_cmvn' in params:
                self.global_mean = torch.from_numpy(np.load(params['global_cmvn'] + '.mean.npy'))
                self.global_std = torch.from_numpy(np.load(params['global_cmvn'] + '.std.npy'))
                logger.info('Load global mean and std vector from files')
                self.apply_global_cmvn = True
            else:
                self.apply_global_cmvn = False

        if self.apply_spec_augment and not self.is_eval:
            self.spec_augment_config = params['spec_augment_config']
            logger.info('Apply SpecAugment!')
            logger.info('Config: %s' % ' '.join([key+':'+str(value) for key, value in self.spec_augment_config.items()]))

        if 'gaussian_noise' in params and not self.is_eval:
            self.gaussian_noise = params['gaussian_noise']
            if self.gaussian_noise > 0.0:
                logger.info('Apply Gaussian Noise with std: %f.' % self.gaussian_noise)
        else:
            self.gaussian_noise = 0.0

        # if 'speed_perturb' in params and not self.is_eval:
        #     self.apply_speed_perturb = params['speed_perturb']
        #     if self.apply_speed_perturb: logger.info('Apply Speed Perturb during the training!')      
        # else:
        #     self.apply_speed_perturb = False

        if 'volume_perturb' in params and not self.is_eval:
            self.apply_volume_perturb = params['volume_perturb']
            if self.apply_volume_perturb: logger.info('Apply Volume Perturb during the training!')
        else:
            self.apply_volume_perturb = False

        self.unit2idx = load_vocab(params['vocab'])

        self.targets_dict = {}
        for text_file in self.datadict['text']:
            with open(text_file, 'r', encoding='utf-8') as t:
                for line in t:
                    parts = line.strip().split()
                    utt_id = parts[0]
                    label = []
                    for c in parts[1:]:
                        label.append(self.unit2idx[c] if c in self.unit2idx else self.unit2idx[UNK_TOKEN])
                    self.targets_dict[utt_id] = label

        self.file_list = []
        for feat_file in self.datadict['feat']:
            with open(feat_file, 'r', encoding='utf-8') as fid:
                for line in fid:
                    idx, path = line.strip().split()
                    self.file_list.append([idx, path])

        assert len(self.file_list) <= len(self.targets_dict)

    def __getitem__(self, index):

        utt_id, path = self.file_list[index]

        if self.feature_extractor in ['torchaudio', 'ta']:
            wavform, sample_frequency = ta.load_wav(path)
        else:
            sample_frequency, wavform = siw.read(path)

        # if self.apply_speed_perturb:
        #     speed_ratio = random.choice([0.9, 1.0, 1.1])
        #     if speed_ratio != 1.0:
        #         wavform = ta.compliance.kaldi.resample_waveform(
        #             wavform, orig_freq=sample_frequency, new_freq=int(sample_frequency*speed_ratio))

        if self.apply_volume_perturb:
            volume_factor = 10 ** (random.uniform(-1.6, 1.6) / 20)
            wavform *= volume_factor

        if self.feature_extractor in ['torchaudio', 'ta']:
            feature = ta.compliance.kaldi.fbank(
                wavform, num_mel_bins=self.params['num_mel_bins'],
                sample_frequency=sample_frequency, dither=0.0
                )
        else:
            feature_np = psf.base.logfbank(wavform, samplerate=sample_frequency, nfilt=self.params['num_mel_bins'])
            feature = torch.FloatTensor(feature_np)

        if self.normalization:
            if self.apply_global_cmvn:
                feature = (feature - self.global_mean) / self.global_std
            else:
                feature = normalization(feature)

        if self.gaussian_noise > 0.0:
            noise = torch.normal(torch.zeros(feature.size(-1)), std=self.gaussian_noise)
            feature += noise

        if self.apply_spec_augment:
            feature = spec_augment(feature)

        feature_length = feature.shape[0]
        targets = self.targets_dict[utt_id]
        targets_length = len(targets)

        return utt_id, feature, feature_length, targets, targets_length

    def __len__(self):
        return len(self.file_list)

    def index_length_pair(self):
        length_list = []
        if 'wav-to-duration' in self.datadict:
            logger.info('Load the wav-to-duration for building buckets!')
            wav_to_duration = {}
            for wav_to_duration_file in self.datadict['wav-to-duration']:
                with open(wav_to_duration_file, 'r') as f:
                    for line in f:
                        uttid, duration = line.strip().split()
                        wav_to_duration[uttid] = duration
            
            for index in range(len(self)):
                uttid, _ = self.file_list[index]
                length_list.append((index, int(float(wav_to_duration[uttid]) * 100)))

        else:
            for index in range(len(self)):
                uttid, path = self.file_list[index]
                wavform, sample_frequency = ta.load_wav(path)
                length_list.append((index, int(wavform.size(1) / sample_frequency * 100)))

        return length_list

    @property
    def idx2unit(self):
        return {i: c for (c, i) in self.unit2idx.items()}

    @property
    def vocab_size(self):
        return len(self.unit2idx)
