import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, d_model, dropout_rate, max_len=5000):
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


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.

    See also: Sec. 3.2  https://arxiv.org/pdf/1809.08895.pdf

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
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


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward

    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate
    """

    def __init__(self, idim, hidden_units, dropout_rate, activation='relu'):
        super(PositionwiseFeedForward, self).__init__()
        self.activation = activation
        self.w_1 = nn.Linear(idim, hidden_units * 2 if activation == 'glu' else hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.w_1(x)
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        elif self.activation == 'glu':
            x = F.glu(x)
        else:
            raise NotImplementedError
        return self.w_2(self.dropout(x))


class LayerNorm(nn.LayerNorm):
    """Layer normalization module

    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization

        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class MultiLayeredConv1d(nn.Module):
    """Multi-layered conv1d for Transformer block.

    This is a module of multi-leyered conv1d designed to replace positionwise feed-forward network
    in Transforner block, which is introduced in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.

    Args:
        in_chans (int): Number of input channels.
        hidden_chans (int): Number of hidden channels.
        kernel_size (int): Kernel size of conv1d.
        dropout_rate (float): Dropout rate.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = nn.Conv1d(in_chans, hidden_chans, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = nn.Conv1d(hidden_chans, in_chans, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, *, in_chans).

        Returns:
            Tensor: Batch of output tensors (B, *, hidden_chans)

        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length)

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(self, idim, odim, dropout_rate):
        super(Conv2dSubsampling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(odim, dropout_rate)
        )

    def forward(self, x, x_mask):
        """Subsample x

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsamplingV2(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length)

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(self, idim, odim, dropout_rate=0.0):
        super(Conv2dSubsamplingV2, self).__init__()

        self.conv1 = nn.Conv2d(1, odim, 3, 2)
        self.conv2 = nn.Conv2d(odim, odim, 3, 2)

        self.linear = nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        self.pos_embedding = PositionalEncoding(odim, dropout_rate)

    def forward(self, inputs, mask):
        """Subsample x

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        inputs = inputs.unsqueeze(1)  # (b, c, t, f)
        inputs = self.conv1(inputs)
        mask = mask[:, :, :-2:2]
        inputs.masked_fill_(~mask.unsqueeze(-1), 0)

        inputs = self.conv2(inputs)
        mask = mask[:, :, :-2:2]
        inputs.masked_fill_(~mask.unsqueeze(-1), 0)

        b, c, t, f = inputs.size()
        inputs = self.linear(inputs.transpose(1, 2).reshape(b, t, c * f))
        encoded_inputs = self.pos_embedding(inputs)

        return inputs, mask


class LinearWithPosEmbedding(nn.Module):
    def __init__(self, input_size, d_model, dropout_rate=0.0):
        super(LinearWithPosEmbedding, self).__init__()
        self.linear = nn.Linear(input_size, d_model)
        # self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        self.pos_embedding = PositionalEncoding(d_model, pos_dropout_rate)

    def forward(self, inputs, mask):

        inputs = self.linear(inputs)
        # inputs = self.norm(inputs)
        inputs = self.activation(self.dropout(inputs))
        
        encoded_inputs = self.pos_embedding(inputs)
        return encoded_inputs, mask
        
