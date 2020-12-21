# File   : utils.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import torch

def get_transformer_decoder_mask(targets):
    batch_size, steps = targets.size()
    seq_mask = torch.ones([batch_size, steps, steps], device=targets.device)
    seq_mask = torch.tril(seq_mask).bool()
    return seq_mask
