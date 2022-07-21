# File   : ffn.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from activation import Swish
import math

logger = logging.getLogger(__name__)


_ACTIVATION = {
    'relu': F.relu,
    'gelu': F.gelu,
    'glu': F.glu,
    'tanh': lambda x: torch.tanh(x),
    'swish': lambda x: x * torch.sigmoid(x)
}


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward
    """

    def __init__(self, d_model, d_ff, dropout, activation='relu'):
        super(PositionwiseFeedForward, self).__init__()
        self.activation = activation

        assert activation in ['relu', 'gelu', 'glu', 'tanh', 'swish']

        self.w_1 = nn.Linear(d_model, d_ff * 2 if activation == 'glu' else d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = _ACTIVATION[self.activation](x)
        return self.w_2(self.dropout(x))


class FeedForwardModule(nn.Module):
    """
    Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.
    Args:
        encoder_dim (int): Dimension of squeezeformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs):
        return self.sequential(inputs)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.
    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int):
        return self.pe[:, :length]


class ResidualConnectionModule(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """
    def __init__(self, module: nn.Module, module_factor: float = 1.0) -> None:
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor

    def forward(self, inputs):
        return (self.module(inputs) * self.module_factor) + inputs


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple) -> None:
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)


def recover_resolution(inputs):
    outputs = list()

    for idx in range(inputs.size(1) * 2):
        outputs.append(inputs[:, idx // 2, :])
    return torch.stack(outputs, dim=1)