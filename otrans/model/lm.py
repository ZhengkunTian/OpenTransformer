import torch
import torch.nn as nn
import torch.nn.functional as F
from otrans.data import PAD
from otrans.metrics import LabelSmoothingLoss
from otrans.layer import TransformerEncoderLayer
from otrans.module import PositionalEncoding


def get_seq_mask(targets):
    batch_size, steps = targets.size()
    seq_mask = torch.ones([batch_size, steps, steps], device=targets.device)
    seq_mask = torch.tril(seq_mask).bool()
    return seq_mask


class TransformerLanguageModel(nn.Module):
    def __init__(self, params):
        super(TransformerLanguageModel, self).__init__()

        self.model_type = 'transformer_lm'
        self.normalize_before = False
        self.smoothing = params['smoothing']
        self.vocab_size = params['vocab_size']
        self.num_blocks = params['num_blocks']

        self.embedding = nn.Embedding(self.vocab_size, params['d_model'])
        self.pos_embedding = PositionalEncoding(params['d_model'], 0.0)

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                params['n_heads'], params['d_model'], params['ffn_units'],
                slf_attn_dropout_rate=0.0, ffn_dropout_rate=0.0,
                residual_dropout_rate=params['residual_dropout_rate'],
                normalize_before=False, concat_after=False, activation='glu') for _ in range(self.num_blocks)
        ])

        if self.normalize_before:
            self.after_norm = nn.LayerNorm(params['d_model'])

        self.output_project = nn.Linear(params['d_model'], self.vocab_size)

        if params['share_embedding']:
            self.output_project.weight = self.embedding.weight
            print('Share the weight of embedding to the output project layer!')

        self.crit = LabelSmoothingLoss(size=self.vocab_size, smoothing=self.smoothing, padding_idx=PAD)

    def forward(self, inputs, targets, pitchs=None):

        dec_mask = get_seq_mask(inputs)
        dec_output = self.embedding(inputs)
        dec_output = self.pos_embedding(dec_output)

        for _, block in enumerate(self.blocks):
            dec_output, dec_mask = block(dec_output, dec_mask)

        if self.normalize_before:
            dec_output = self.after_norm(dec_output)

        logits = self.output_project(dec_output)
        loss = self.crit(logits, targets)

        return loss

    def predict(self, targets):

        dec_output = self.embedding(targets)
        dec_output = self.pos_embedding(dec_output)

        dec_mask = get_seq_mask(targets)

        for _, block in enumerate(self.blocks):
            dec_output, dec_mask = block(dec_output, dec_mask)

        if self.normalize_before:
            dec_output = self.after_norm(dec_output)

        logits = self.output_project(dec_output)

        log_probs = F.log_softmax(logits[:, -1, :].unsqueeze(1), dim=-1)
        return log_probs
