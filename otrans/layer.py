import torch
import torch.nn as nn
import torch.nn.functional as F
from otrans.attention import MultiHeadedAttention
from otrans.module import LayerNorm, PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    def __init__(self, attention_heads, d_model, linear_units, slf_attn_dropout_rate, 
                 ffn_dropout_rate, residual_dropout_rate, normalize_before=False,
                 concat_after=False, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(attention_heads, d_model, slf_attn_dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, linear_units, ffn_dropout_rate, activation)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(residual_dropout_rate)
        self.dropout2 = nn.Dropout(residual_dropout_rate)

        self.normalize_before = normalize_before
        self.concat_after = concat_after

        if self.concat_after:
            self.concat_linear = nn.Linear(d_model * 2, d_model)

    def forward(self, x, mask):
        """Compute encoded features

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout1(self.self_attn(x, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout2(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask


class TransformerDecoderLayer(nn.Module):

    def __init__(self, attention_heads, d_model, linear_units, slf_attn_dropout_rate, src_attn_dropout_rate, 
                 ffn_dropout_rate, residual_dropout_rate, normalize_before=True, concat_after=False, activation='relu'):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(attention_heads, d_model, slf_attn_dropout_rate)
        self.src_attn = MultiHeadedAttention(attention_heads, d_model, src_attn_dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, linear_units, ffn_dropout_rate, activation)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(residual_dropout_rate)
        self.dropout2 = nn.Dropout(residual_dropout_rate)
        self.dropout3 = nn.Dropout(residual_dropout_rate)

        self.normalize_before = normalize_before
        self.concat_after = concat_after

        if self.concat_after:
            self.concat_linear1 = nn.Linear(d_model * 2, d_model)
            self.concat_linear2 = nn.Linear(d_model * 2, d_model)

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Compute decoded features

        :param torch.Tensor tgt: decoded previous target features (batch, max_time_out, size)
        :param torch.Tensor tgt_mask: mask for x (batch, max_time_out)
        :param torch.Tensor memory: encoded source features (batch, max_time_in, size)
        :param torch.Tensor memory_mask: mask for memory (batch, max_time_in)
        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if self.concat_after:
            tgt_concat = torch.cat((tgt, self.self_attn(tgt, tgt, tgt, tgt_mask)), dim=-1)
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout1(self.self_attn(tgt, tgt, tgt, tgt_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.cat((x, self.src_attn(x, memory, memory, memory_mask)), dim=-1)
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout2(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout3(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        return x, tgt_mask


