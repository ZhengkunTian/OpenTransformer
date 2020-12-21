# File   : attention.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import math
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BasedAttention(nn.Module):
    def __init__(self, source_dim, output_dim, enable_output_proj=True, dropout=0.0):
        super(BasedAttention, self).__init__()

        self.enable_output_proj = enable_output_proj
        if self.enable_output_proj:
            self.output_proj = nn.Linear(source_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def compute_context(self, values, scores, mask=None):
        """
        Args:
            values: [b, t2, v] or [b, nh, t2, v]
            scores: [b, t1, t2] or [b, nh, t1, t2]
            mask: [b, t1, t2] or [b, 1/nh, t1, t2]
        """

        assert values.dim() == scores.dim()

        if mask is not None:
            scores.masked_fill_(~mask, -float('inf'))
        
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, values)

        if context.dim() == 4:
            b, n, t, v = context.size()
            context = context.transpose(1, 2).reshape(b, t, n * v)
        
        if self.enable_output_proj:
            context = self.output_proj(context)

        return self.dropout(context), weights


class MultiHeadedSelfAttention(BasedAttention):
    def __init__(self, n_heads, d_model, dropout_rate=0.0, share_qvk_proj=False):
        super(MultiHeadedSelfAttention, self).__init__(d_model, d_model, enable_output_proj=True, dropout=dropout_rate)

        self.d_model = d_model
        self.share_qvk_proj = share_qvk_proj
        self.nheads = n_heads
        self.d_k = d_model // n_heads

        self.qvk_proj = nn.Linear(d_model, d_model if self.share_qvk_proj else d_model * 3)

    def forward(self, x, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor mask: (batch, time1 or 1, time2)
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
        """

        x = self.qvk_proj(x)

        if self.share_qvk_proj:
            query = key = value = x
        else:
            query, key, value = torch.split(x, self.d_model, dim=-1)

        batch_size = x.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        context, attn_weights = self.compute_context(value, scores, mask.unsqueeze(1) if mask is not None else None)

        return context, attn_weights

    def inference(self, x, mask, cache=None):

        x = self.qvk_proj(x)

        if self.share_qvk_proj:
            query = key = value = x
        else:
            query, key, value = torch.split(x, self.d_model, dim=-1)

        batch_size = x.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        context, attn_weights = self.compute_context(value, scores, mask.unsqueeze(1) if mask is not None else None)

        return context, attn_weights, cache


class MultiHeadedCrossAttention(BasedAttention):
    def __init__(self, n_heads, d_model, memory_dim, dropout_rate=0.0, share_vk_proj=False):
        super(MultiHeadedCrossAttention, self).__init__(d_model, d_model, enable_output_proj=True, dropout=dropout_rate)

        self.d_model = d_model
        self.share_vk_proj = share_vk_proj
        self.nheads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.vk_proj = nn.Linear(memory_dim, d_model if self.share_vk_proj else d_model * 2)

    def forward(self, query, memory, memory_mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor memory: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1 or 1, time2)
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
        """

        query = self.q_proj(query)
        memory = self.vk_proj(memory)

        if self.share_vk_proj:
            key = value = memory
        else:
            key, value = torch.split(memory, self.d_model, dim=-1)

        batch_size = query.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        context, attn_weights = self.compute_context(value, scores, memory_mask.unsqueeze(1))

        return context, attn_weights

    def inference(self, query, memory, memory_mask, cache=None):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor memory: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1 or 1, time2)
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
        """

        query = self.q_proj(query)
        memory = self.vk_proj(memory)

        if self.share_vk_proj:
            key = value = memory
        else:
            key, value = torch.split(memory, self.d_model, dim=-1)

        batch_size = query.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        context, attn_weights = self.compute_context(value, scores, memory_mask.unsqueeze(1))

        return context, attn_weights, cache


class MultiHeadedSelfAttentionWithRelPos(MultiHeadedSelfAttention):
    def __init__(self, n_heads, d_model, dropout_rate=0.0, share_qvk_proj=False):
        super(MultiHeadedSelfAttentionWithRelPos, self).__init__(n_heads, d_model, dropout_rate, share_qvk_proj)

        self.pos_proj = nn.Linear(d_model, d_model, bias=False)
        self.posu = nn.Parameter(torch.Tensor(n_heads, self.d_k))
        self.posv = nn.Parameter(torch.Tensor(n_heads, self.d_k))

        torch.nn.init.xavier_normal_(self.posu)
        torch.nn.init.xavier_normal_(self.posv)

    def _shift(self, x, zero_triu=False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, nheads, time, size).
            zero_triu (bool): If true, return the lower triangular part of the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """
        b, nh, t, v = x.size()
        zero_pad = torch.zeros((b, nh, t, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1) # (b, nh, t, v+1)

        x_padded = x_padded.view(b, nh, v+1, t) # (b, nh, v+1, t)
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((t, v))
            x = x * torch.tril(ones, v - t)[None, None, :, :]

        return x

    def forward(self, x, mask, pos):
        """
        Args:
            x: [batch_size, time, size]
            mask: [batch_size, 1, time]
            pos: positional embedding [batch_size, time, size]
        """

        x = self.qvk_proj(x)

        if self.share_qvk_proj:
            query = key = value = x
        else:
            query, key, value = torch.split(x, self.d_model, dim=-1)

        batch_size = x.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)

        bpos = pos.size(0)
        pos = self.pos_proj(pos).reshape(bpos, -1, self.nheads, self.d_k).transpose(1, 2)

        query_with_bias_u = query + self.posu
        query_with_bias_u = query_with_bias_u.transpose(1, 2)

        query_with_bias_v = query + self.posv
        query_with_bias_v = query_with_bias_v.transpose(1, 2)

        matrix_ac = torch.matmul(query_with_bias_u, key.transpose(-2, -1))

        matrix_bd = torch.matmul(query_with_bias_v, pos.transpose(-2, -1))
        matrix_bd = self._shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        context, attn_weights = self.compute_context(value, scores, mask.unsqueeze(1) if mask is not None else None)

        return context, attn_weights


    def inference(self, x, mask, pos, cache):
        """
        Args:
            x: [batch_size, time, size]
            mask: [batch_size, 1, time]
            pos: positional embedding [batch_size, time, size]
        """

        x = self.qvk_proj(x)

        if self.share_qvk_proj:
            query = key = value = x
        else:
            query, key, value = torch.split(x, self.d_model, dim=-1)

        batch_size = x.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)

        bpos = pos.size(0)
        pos = self.pos_proj(pos).reshape(bpos, -1, self.nheads, self.d_k).transpose(1, 2)

        query_with_bias_u = query + self.posu
        query_with_bias_u = query_with_bias_u.transpose(1, 2)

        query_with_bias_v = query + self.posv
        query_with_bias_v = query_with_bias_v.transpose(1, 2)

        matrix_ac = torch.matmul(query_with_bias_u, key.transpose(-2, -1))

        matrix_bd = torch.matmul(query_with_bias_v, pos.transpose(-2, -1))
        matrix_bd = self._shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        context, attn_weights = self.compute_context(value, scores, mask.unsqueeze(1))

        return context, attn_weights, cache
