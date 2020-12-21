# File   : conformer.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com


import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from otrans.encoder.base import BaseEncoder
from otrans.module.ffn import PositionwiseFeedForward
from otrans.module.attention import MultiHeadedSelfAttentionWithRelPos, MultiHeadedSelfAttention
from otrans.module.conformer import ConformerConvolutionModule
from otrans.module.pos import MixedPositionalEncoding, RelPositionalEncoding


logger = logging.getLogger(__name__)


class ConformerEncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, cov_kernel_size, n_heads, slf_attn_dropout=0.0, ffn_dropout=0.0,
                 residual_dropout=0.1, conv_dropout=0.0, macaron_style=True, ffn_scale=0.5, conv_bias=True,
                 relative_positional=True, activation='glu'):
        super(ConformerEncoderBlock, self).__init__()

        self.macaron_style = macaron_style
        self.ffn_scale = ffn_scale
        self.relative_positional = relative_positional
        self.residual_dropout = residual_dropout

        if self.macaron_style:
            self.pre_ffn = PositionwiseFeedForward(d_model, d_ff, ffn_dropout, activation=activation)
            self.macaron_ffn_norm = nn.LayerNorm(d_model)

        if self.relative_positional:
            self.mha = MultiHeadedSelfAttentionWithRelPos(n_heads, d_model, slf_attn_dropout)
        else:
            self.mha = MultiHeadedSelfAttention(n_heads, d_model, slf_attn_dropout)
        self.mha_norm = nn.LayerNorm(d_model)

        self.conv = ConformerConvolutionModule(d_model, cov_kernel_size, conv_bias, conv_dropout)
        self.conv_norm = nn.LayerNorm(d_model)

        self.post_ffn = PositionwiseFeedForward(d_model, d_ff, ffn_dropout, activation=activation)
        self.post_ffn_norm = nn.LayerNorm(d_model)

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask, pos=None):

        if self.macaron_style:
            residual = x
            x = self.macaron_ffn_norm(x)
            x = residual + self.ffn_scale * F.dropout(self.pre_ffn(x), p=self.residual_dropout)

        residual = x
        x = self.mha_norm(x)
        if self.relative_positional:
            slf_attn_out, slf_attn_weights = self.mha(x, mask.unsqueeze(1), pos)
        else:
            slf_attn_out, slf_attn_weights = self.mha(x, mask.unsqueeze(1))
        x = residual + F.dropout(slf_attn_out, p=self.residual_dropout)

        residual = x
        x = self.conv_norm(x)
        x = residual + F.dropout(self.conv(x, mask), p=self.residual_dropout)

        residual = x
        x = self.post_ffn_norm(x)
        x = residual + self.ffn_scale * F.dropout(self.post_ffn(x), p=self.residual_dropout)

        x = self.final_norm(x)

        return x, {'slf_attn_weights': slf_attn_weights}

    def inference(self, x, mask, pos=None, cache=None):

        if self.macaron_style:
            residual = x
            x = self.macaron_ffn_norm(x)
            x = residual + self.ffn_scale * self.pre_ffn(x)

        residual = x
        x = self.mha_norm(x)
        if self.relative_positional:
            slf_attn_out, slf_attn_weights, new_cache = self.mha.inference(x, mask.unsqueeze(1), pos, cache)
        else:
            slf_attn_out, slf_attn_weights, new_cache = self.mha.inference(x, mask.unsqueeze(1), cache)
        x = residual + slf_attn_out

        residual = x
        x = self.conv_norm(x)
        x = residual + self.conv(x, mask)

        residual = x
        x = self.post_ffn_norm(x)
        x = residual + self.ffn_scale * self.post_ffn(x)

        x = self.final_norm(x)

        return x, new_cache, {'slf_attn_weights': slf_attn_weights}


class ConformerEncoder(BaseEncoder):
    def __init__(self, d_model, d_ff, cov_kernel_size, n_heads, nblocks=12, pos_dropout=0.0,
                 slf_attn_dropout=0.0, ffn_dropout=0.0, residual_dropout=0.1, conv_dropout=0.0, macaron_style=True,
                 ffn_scale=0.5, conv_bias=True, relative_positional=True, activation='glu'):
        super(ConformerEncoder, self).__init__()

        self.relative_positional = relative_positional

        if self.relative_positional:
            self.posemb = RelPositionalEncoding(d_model, pos_dropout)
        else:
            self.posemb = MixedPositionalEncoding(d_model, pos_dropout)

        self.blocks = nn.ModuleList(
            [
                ConformerEncoderBlock(
                    d_model, d_ff, cov_kernel_size, n_heads, slf_attn_dropout, ffn_dropout, residual_dropout,
                    conv_dropout, macaron_style, ffn_scale, conv_bias, relative_positional, activation
                ) for _ in range(nblocks)
            ]
        )

        self.output_size = d_model

    def forward(self, x, mask):
        
        x, pos = self.posemb(x)

        x.masked_fill_(~mask.unsqueeze(2), 0.0)

        attn_weights = {}
        for i, block in enumerate(self.blocks):
            x, attn_weight = block(x, mask, pos)
            attn_weights['enc_block_%d' % i] = attn_weight

        return x, mask, attn_weights

    def inference(self, x, mask, cache=None):

        x, pos = self.posemb.inference(x)

        x.masked_fill_(~mask.unsqueeze(2), 0.0)

        attn_weights = {}
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, new_cache, attn_weight = block.inference(x, mask, pos, cache[i] if isinstance(cache, list) else cache)
            new_caches.append(new_cache)
            attn_weights['enc_block_%d' % i] = attn_weight

        return x, mask, new_caches, attn_weights