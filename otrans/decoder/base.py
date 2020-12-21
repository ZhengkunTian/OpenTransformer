# File   : base.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import torch
import torch.nn as nn


class BaseDecoder(nn.Module):
    def __init__(self):
        super(BaseDecoder, self).__init__()

    def forward(self, targets, **kwargs):
        raise NotImplementedError

    def inference(self, tokens, **kargs):
        raise NotImplementedError