# File   : utils.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import torch
import torch.nn as nn

def get_length_mask(tensor, tensor_length):
    b, t, _ = tensor.size()  
    mask = tensor.new_zeros([b, t], dtype=torch.uint8)
    for i, length in enumerate(tensor_length):
        length = length.item()
        mask[i].narrow(0, 0, length).fill_(1)
    return mask > 0

