import torch
import torch.nn as nn
from otrans.encoder import TransformerEncoder
from otrans.decoder import TransformerDecoder
from otrans.metrics import LabelSmoothingLoss
from otrans.utils import initialize


class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()

        self.params = params

        self.encoder = TransformerEncoder(input_size=params['feat_dim'],
                                          d_model=params['d_model'],
                                          attention_heads=params['n_heads'],
                                          linear_units=params['enc_ffn_units'],
                                          num_blocks=params['num_enc_blocks'],
                                          pos_dropout_rate=params['pos_dropout_rate'],
                                          slf_attn_dropout_rate=params['slf_attn_dropout_rate'],
                                          ffn_dropout_rate=params['ffn_dropout_rate'],
                                          residual_dropout_rate=params['residual_dropout_rate'],
                                          input_layer=params['enc_input_layer'],
                                          normalize_before=params['normalize_before'],
                                          concat_after=params['concat_after'],
                                          activation=params['activation'])

        self.decoder = TransformerDecoder(output_size=params['vocab_size'],
                                          d_model=params['d_model'],
                                          attention_heads=params['n_heads'],
                                          linear_units=params['dec_ffn_units'],
                                          num_blocks=params['num_dec_blocks'],
                                          pos_dropout_rate=params['pos_dropout_rate'],
                                          slf_attn_dropout_rate=params['slf_attn_dropout_rate'],
                                          src_attn_dropout_rate=params['src_attn_dropout_rate'],
                                          ffn_dropout_rate=params['ffn_dropout_rate'],
                                          residual_dropout_rate=params['residual_dropout_rate'],
                                          normalize_before=params['normalize_before'],
                                          concat_after=params['concat_after'],
                                          activation=params['activation'],
                                          share_embedding=params['share_embedding'])

        self.crit = LabelSmoothingLoss(size=params['vocab_size'],
                                       smoothing=params['smoothing'])

    def forward(self, inputs, inputs_length, targets, targets_length):

        # 1. forward encoder
        enc_states, enc_mask = self.encoder(inputs, inputs_length)

        # 2. forward decoder
        target_in = targets[:, :-1].clone()
        logits, _ = self.decoder(target_in, targets_length, enc_states, enc_mask)

        # 3. compute attention loss
        target_out = targets[:, 1:].clone()
        loss = self.crit(logits, target_out)

        return loss

