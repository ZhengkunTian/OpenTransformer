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
from otrans.module.pos import PositionalEncoding


logger = logging.getLogger(__name__)


class ConformerEncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, cov_kernel_size, n_heads, slf_attn_dropout=0.0, ffn_dropout=0.0,
                 residual_dropout=0.1, conv_dropout=0.0, macaron_style=True, conv_first=False,
                 ffn_scale=0.5, conv_bias=True, relative_positional=True, activation='glu'):
        super(ConformerEncoderBlock, self).__init__()

        self.conv_first = conv_first
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

    def pre_ffn_forward(self, x, dropout=0.0):
        residual = x
        x = self.macaron_ffn_norm(x)
        return residual + self.ffn_scale * F.dropout(self.pre_ffn(x), p=dropout)
    
    def pos_ffn_forward(self, x, dropout=0.0):
        residual = x
        x = self.post_ffn_norm(x)
        return residual + self.ffn_scale * F.dropout(self.post_ffn(x), p=dropout)

    def conv_augment_forward(self, x, mask, dropout=0.0):
        residual = x
        x = self.conv_norm(x)
        return residual + F.dropout(self.conv(x, mask), p=dropout)

    def attn_forward(self, x, mask, pos, dropout=0.0):
        residual = x
        x = self.mha_norm(x)
        if self.relative_positional:
            slf_attn_out, slf_attn_weights = self.mha(x, mask.unsqueeze(1), pos)
        else:
            slf_attn_out, slf_attn_weights = self.mha(x, mask.unsqueeze(1))
        slf_attn_out = residual + F.dropout(slf_attn_out, p=dropout)
        return slf_attn_out, slf_attn_weights

    def forward(self, x, mask, pos=None):

        if self.macaron_style:
            x = self.pre_ffn_forward(x, dropout=self.residual_dropout)

        if self.conv_first:
            x = self.conv_augment_forward(x, mask, dropout=self.residual_dropout)
            x, slf_attn_weights = self.attn_forward(x, mask, pos, dropout=self.residual_dropout)
        else:
            x, slf_attn_weights = self.attn_forward(x, mask, pos, dropout=self.residual_dropout)
            x = self.conv_augment_forward(x, mask, dropout=self.residual_dropout)

        x = self.post_ffn_norm(x)

        return self.final_norm(x), {'slf_attn_weights': slf_attn_weights}

    def attn_infer(self, x, mask, pos, cache):
        residual = x
        x = self.mha_norm(x)
        if self.relative_positional:
            slf_attn_out, slf_attn_weights, new_cache = self.mha.inference(x, mask.unsqueeze(1), pos, cache)
        else:
            slf_attn_out, slf_attn_weights, new_cache = self.mha.inference(x, mask.unsqueeze(1), cache)
        return residual + slf_attn_out, slf_attn_weights, new_cache

    def inference(self, x, mask, pos=None, cache=None):

        if self.macaron_style:
            x = self.pre_ffn_forward(x)

        if self.conv_first:
            x = self.conv_augment_forward(x, mask)
            x, slf_attn_weights, new_cache = self.attn_infer(x, mask, pos, cache)
        else:
            x, slf_attn_weights, new_cache = self.attn_infer(x, mask, pos, cache)
            x = self.conv_augment_forward(x, mask)

        x = self.post_ffn_norm(x)

        return self.final_norm(x),  new_cache, {'slf_attn_weights': slf_attn_weights}


class ConformerEncoder(BaseEncoder):
    def __init__(self, d_model, d_ff, cov_kernel_size, n_heads, nblocks=12, pos_dropout=0.0,
                 slf_attn_dropout=0.0, ffn_dropout=0.0, residual_dropout=0.1, conv_dropout=0.0, macaron_style=True,
                 ffn_scale=0.5, conv_bias=True, positional_encoding=True, relative_positional=True, conv_first=False, activation='glu'):
        super(ConformerEncoder, self).__init__()

        self.positional_encoding = positional_encoding
        self.relative_positional = relative_positional
        self.output_size = d_model

        if self.positional_encoding:
            self.pos_emb = PositionalEncoding(d_model, pos_dropout)

        self.blocks = nn.ModuleList(
            [
                ConformerEncoderBlock(
                    d_model, d_ff, cov_kernel_size, n_heads, slf_attn_dropout, ffn_dropout, residual_dropout,
                    conv_dropout, macaron_style, conv_first, ffn_scale, conv_bias, relative_positional, activation
                ) for _ in range(nblocks)
            ]
        )

        self.output_size = d_model

    def _pos_encoding(self, inputs):
        if self.relative_positional:
            enc_output = inputs
            # [1, 2T - 1]
            position = torch.arange(-(inputs.size(1)-1), inputs.size(1), device=inputs.device).reshape(1, -1)
            pos = self.pos_emb._embedding_from_positions(position)
        else:  
            enc_output, pos = self.pos_emb(inputs)
        return enc_output, pos

    def forward(self, inputs, mask):

        if self.positional_encoding:
            enc_output, pos = self._pos_encoding(inputs)
        else:
            enc_output = inputs
            pos = None
        
        attn_weights = {}
        for i, block in enumerate(self.blocks):
            enc_output, attn_weight = block(enc_output, mask, pos)
            attn_weights['enc_block_%d' % i] = attn_weight

        return enc_output, mask, attn_weights

    def inference(self, inputs, mask, cache=None):

        if self.positional_encoding:
            enc_output, pos = self._pos_encoding(inputs)
        else:
            enc_output = inputs
            pos = None

        # x.masked_fill_(~mask.unsqueeze(2), 0.0)

        attn_weights = {}
        new_caches = []
        for i, block in enumerate(self.blocks):
            enc_output, new_cache, attn_weight = block.inference(enc_output, mask, pos, cache[i] if isinstance(cache, list) else cache)
            new_caches.append(new_cache)
            attn_weights['enc_block_%d' % i] = attn_weight

        return enc_output, mask, new_caches, attn_weights
