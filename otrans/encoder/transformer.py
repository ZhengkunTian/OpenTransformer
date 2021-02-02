# File   : transformer.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import logging
import torch
import torch.nn as nn
from otrans.module.pos import PositionalEncoding
from otrans.module.ffn import PositionwiseFeedForward
from otrans.module.attention import MultiHeadedSelfAttention, MultiHeadedSelfAttentionWithRelPos


logger = logging.getLogger(__name__)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, slf_attn_dropout, ffn_dropout, residual_dropout,
                 normalize_before=False, concat_after=False, relative_positional=False, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()

        self.relative_positional = relative_positional

        if self.relative_positional:
            self.slf_attn = MultiHeadedSelfAttentionWithRelPos(n_heads, d_model, slf_attn_dropout)
        else:
            self.slf_attn = MultiHeadedSelfAttention(n_heads, d_model, slf_attn_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, ffn_dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(residual_dropout)
        self.dropout2 = nn.Dropout(residual_dropout)

        self.normalize_before = normalize_before
        self.concat_after = concat_after

        if self.concat_after:
            self.concat_linear = nn.Linear(d_model * 2, d_model)

    def forward(self, x, mask, pos=None):
        if self.normalize_before:
            x = self.norm1(x)
        residual = x

        if self.relative_positional:
            slf_attn_out, slf_attn_weights = self.slf_attn(x, mask, pos)
        else:
            slf_attn_out, slf_attn_weights = self.slf_attn(x, mask)
    
        if self.concat_after:
            x = residual + self.concat_linear(torch.cat((x, slf_attn_out), dim=-1))
        else:
            x = residual + self.dropout1(slf_attn_out)
        if not self.normalize_before:
            x = self.norm1(x)

        if self.normalize_before:
            x = self.norm2(x)
        residual = x
        x = residual + self.dropout2(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, {'slf_attn_weights': slf_attn_weights}

    def inference(self, x, mask, pos=None, cache=None):
        if self.normalize_before:
            x = self.norm1(x)
        residual = x
        if self.relative_positional:
            slf_attn_out, slf_attn_weights, new_cache = self.slf_attn.inference(x, mask, cache, pos)
        else:
            slf_attn_out, slf_attn_weights, new_cache = self.slf_attn.inference(x, mask, cache)

        if self.concat_after:
            x = residual + self.concat_linear(torch.cat((x, slf_attn_out), dim=-1))
        else:
            x = residual + slf_attn_out
        if not self.normalize_before:
            x = self.norm1(x)

        if self.normalize_before:
            x = self.norm2(x)
        residual = x
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)

        return x, new_cache, {'slf_attn_weights': slf_attn_weights}


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=2048, n_blocks=6, pos_dropout=0.0, 
                 slf_attn_dropout=0.0, ffn_dropout=0.0, residual_dropout=0.1, normalize_before=False,
                 concat_after=False, relative_positional=False, activation='relu'):
        super(TransformerEncoder, self).__init__()

        self.normalize_before = normalize_before
        self.relative_positional = relative_positional

        self.pos_emb = PositionalEncoding(d_model, pos_dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                n_heads, d_model, d_ff, slf_attn_dropout, ffn_dropout,
                residual_dropout=residual_dropout, normalize_before=normalize_before,
                concat_after=concat_after, relative_positional=relative_positional, activation=activation) for _ in range(n_blocks)
        ])

        if self.normalize_before:
            self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs, mask):
    
        if self.relative_positional:
            enc_output = inputs
            # [1, 2T - 1]
            position = torch.arange(-(inputs.size(1)-1), inputs.size(1), device=inputs.device).reshape(1, -1)
            pos = self.pos_emb._embedding_from_positions(position)
        else:  
            enc_output, pos = self.pos_emb(inputs)

        # enc_output.masked_fill_(~mask.unsqueeze(2), 0.0)

        attn_weights = {}
        for i, block in enumerate(self.blocks):
            enc_output, attn_weight = block(enc_output, mask.unsqueeze(1), pos)
            attn_weights['enc_block_%d' % i] = attn_weight

        if self.normalize_before:
            enc_output = self.norm(enc_output)

        return enc_output, mask, attn_weights

    # def inference(self, inputs, mask, cache=None):
    
    #     enc_output, pos = self.pos_emb.inference(inputs)

    #     enc_output.masked_fill_(~mask.unsqueeze(2), 0.0)

    #     attn_weights = {}
    #     new_caches = []
    #     for i, block in enumerate(self.blocks):
    #         enc_output, new_cache, attn_weight = block.inference(enc_output, mask.unsqueeze(1), pos, cache)
    #         attn_weights['enc_block_%d' % i] = attn_weight
    #         new_caches.append(new_cache)

    #     if self.normalize_before:
    #         enc_output = self.norm(enc_output)

    #     return enc_output, mask, new_caches, attn_weights

