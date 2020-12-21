# File   : concat.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import torch
import torch.nn as nn
import torch.nn.functional as F
from otrans.frontend.base import BaseFrontEnd


class ConcatFeatureFrontEnd(BaseFrontEnd):
    def __init__(self, input_size, left_frames, right_frames, frame_rate=30):
        super(ConcatFeatureFrontEnd, self).__init__()

        self.left_frames = left_frames
        self.right_frames = right_frames
        self.frame_rate = frame_rate

        self.stride = int(self.frame_rate / 10)
        self.input_size = input_size
        self.nframes = self.left_frames + self.right_frames + 1
        self.output_size = self.nframes * input_size

        self.window = nn.Unfold(kernel_size=(self.nframes, self.input_size), stride=self.stride, padding=0)

    def forward(self, x, mask):

        feat_len = x.size(1)
        if (feat_len - self.nframes) % self.stride != 0:
            pad_len = self.stride - (feat_len - self.nframes) % self.stride
            x = F.pad(x, pad=(0, 0, 0, pad_len), value=0.0)
            mask = F.pad(mask.int(), pad=(0, pad_len), value=0) > 0
        else:
            pad_len = 0

        with torch.no_grad():
            x = self.window(x.unsqueeze(1))
            x = x.transpose(1, 2)
            
        mask = mask[:, self.left_frames::self.stride]
        assert mask.size(1) == x.size(1)

        return x, mask
        

class ConcatWithLinearFrontEnd(ConcatFeatureFrontEnd):
    def __init__(self, input_size, output_size, left_frames, right_frames, frame_rate=30, dropout=0.0):
        super().__init__(input_size, left_frames, right_frames, frame_rate)

        self.linear = nn.Linear(self.output_size, output_size)
        self.dropout = dropout

    def forward(self, x, mask):
        x, mask = super().forward(x, mask)
        return F.dropout(self.linear(x), p=self.dropout), mask
