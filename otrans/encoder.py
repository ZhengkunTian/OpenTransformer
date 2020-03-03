import torch
import torch.nn as nn
import torch.nn.functional as F
from otrans.module import *
from otrans.utils import get_enc_padding_mask
from otrans.layer import TransformerEncoderLayer


class TransformerEncoder(nn.Module):

    def __init__(self, input_size, d_model=256, attention_heads=4, linear_units=2048, num_blocks=6, pos_dropout_rate=0.0,
                 slf_attn_dropout_rate=0.0, ffn_dropout_rate=0.0, residual_dropout_rate=0.1, input_layer="conv2d",
                 normalize_before=True, concat_after=False, activation='relu', type='transformer'):
        super(TransformerEncoder, self).__init__()

        self.normalize_before = normalize_before

        if input_layer == "linear":
            self.embed = LinearWithPosEmbedding(input_size, d_model, pos_dropout_rate)
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, d_model, pos_dropout_rate)
        elif input_layer == 'conv2dv2':
            self.embed = Conv2dSubsamplingV2(input_size, d_model, pos_dropout_rate)

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(attention_heads, d_model, linear_units, slf_attn_dropout_rate, ffn_dropout_rate,
                                    residual_dropout_rate=residual_dropout_rate, normalize_before=normalize_before,
                                    concat_after=concat_after, activation=activation) for _ in range(num_blocks)
        ])

        if self.normalize_before:
            self.after_norm = LayerNorm(d_model)

    def forward(self, inputs, input_length):

        enc_mask = get_enc_padding_mask(inputs, input_length)
        enc_output, enc_mask = self.embed(inputs, enc_mask)

        enc_output.masked_fill_(~enc_mask.transpose(1, 2), 0.0)

        for _, block in enumerate(self.blocks):
            enc_output, enc_mask = block(enc_output, enc_mask)
            enc_output.masked_fill_(~enc_mask.transpose(1, 2), 0.0)

        if self.normalize_before:
            enc_output = self.after_norm(enc_output)

        return enc_output, enc_mask



