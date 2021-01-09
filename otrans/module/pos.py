# File   : pos.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com


import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, d_model, dropout_rate=0.0, max_len=5000):
        """Initialize class.

        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length

        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self.mid_len = int(max_len//2)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * 
                             -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.size(1)]
        return self.dropout(x)

    def inference(self, x, startid=0):
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, startid:startid+x.size(1)]
        return x, None


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.

    See also: Sec. 3.2  https://arxiv.org/pdf/1809.08895.pdf

    """

    def __init__(self, d_model, dropout_rate=0.0, max_len=5000):
        """Initialize class.

        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length

        """
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, :x.size(1)]
        return self.dropout(x)

    def inference(self, x, startid=0):
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, startid:startid+x.size(1)]
        return x, None  


class MixedPositionalEncoding(PositionalEncoding):
    """Mixed Scaled positional encoding module.

        Two Modes:
            Default scale and learnable scale!

    """

    def __init__(self, d_model, dropout_rate=0.0, max_len=5000, scale_learnable=False):
        """Initialize class.

        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length

        """
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)

        self.scale_learnable = scale_learnable

        if self.scale_learnable:
            self.alpha = nn.Parameter(torch.tensor(1.0))
           
    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        self.extend_pe(x)
        if self.scale_learnable:
            x = x + self.alpha * self.pe[:, :x.size(1)]
        else:
            x = x * self.xscale + self.pe[:, :x.size(1)]
        return self.dropout(x), None

    def inference(self, x, startid=0):
        self.extend_pe(x)
        if self.scale_learnable:
            x = x + self.alpha * self.pe[:, startid:startid+x.size(1)]
        else:
            x = x * self.xscale + self.pe[:, startid:startid+x.size(1)]
        return x, None


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model, dropout_rate=0.0, max_len=5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len)

    def forward(self, x):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale
        ipt_length = x.size(1)
        # set position 0 at self.mid_len
        # relative position has range [-ipt_len+1, ipt_len]
        pos_emb = self.pe[:, self.mid_len-ipt_length+1: self.mid_len+ipt_length]
        return self.dropout(x), self.dropout(pos_emb)
