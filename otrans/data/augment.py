# File   : augment.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import random
import numpy as np


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


