import torch
import torch.nn as nn
import torch.nn.functional as F
from otrans.data import PAD
from otrans.module.loss import LabelSmoothingLoss
from otrans.encoder.transformer import TransformerEncoderLayer
from otrans.module.pos import PositionalEncoding


LOGZERO = -10000000000.0
ZERO = 1.0e-10


def get_seq_mask(targets):
    batch_size, steps = targets.size()
    seq_mask = torch.ones([batch_size, steps, steps], device=targets.device)
    seq_mask = torch.tril(seq_mask).bool()
    return seq_mask


class BaseLM(nn.Module):
    def __init__(self, params):
        super(BaseLM, self).__init__()
        self.params = params

    def forward(self, inputs, targets):
        raise NotImplementedError

    def set_epoch(self, epoch):
        pass


class RecurrentLanguageModel(BaseLM):
    def __init__(self, params):
        super(RecurrentLanguageModel, self).__init__(params)

        self.model_type = 'recurrent_lm'
        self.vocab_size = params['vocab_size']
        self.share_embedding = params['share_embedding']
        self.smoothing = params['smoothing']
        self.num_layers = params['num_layers']
        self.hidden_size = params['hidden_size']

        self.embedding = nn.Embedding(params['vocab_size'], params['hidden_size'])
        self.rnn = nn.LSTM(input_size=params['hidden_size'],
                           hidden_size=params['hidden_size'],
                           num_layers=params['num_layers'],
                           batch_first=True,
                           dropout=params['dropout'],
                           bidirectional=False)

        self.output_project = nn.Linear(
            params['hidden_size'], params['vocab_size'])

        if self.share_embedding:
            assert self.embedding.weight.size() == self.output_project.weight.size()
            self.output_project.weight = self.embedding.weight

        self.crit = LabelSmoothingLoss(size=self.vocab_size, smoothing=self.smoothing, 
                                       padding_idx=PAD)

    def forward(self, inputs, targets):

        emb_inputs = self.embedding(inputs['inputs'])

        self.rnn.flatten_parameters()
        outputs, _ = self.rnn(emb_inputs)

        logits = self.output_project(outputs)

        loss = self.crit(logits, targets['targets'])
        return loss, None

    def predict(self, pred, hidden=None):

        emb_inputs = self.embedding(pred)
        outputs, hidden = self.rnn(emb_inputs, hidden)
        logits = self.output_project(outputs)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, hidden

    def save_checkpoint(self, params, name):
        checkpoint = {
            'params': params,
            'model': self.state_dict()
            }

        torch.save(checkpoint, name)

    def init_hidden_states(self, batch_size, device):
        return torch.zeros([self.num_layers, batch_size, self.hidden_size]).to(device)


class TransformerLanguageModel(BaseLM):
    def __init__(self, params):
        super(TransformerLanguageModel, self).__init__(params)

        self.model_type = 'transformer_lm'
        self.normalize_before = False
        self.smoothing = params['smoothing']
        self.vocab_size = params['vocab_size']
        self.num_blocks = params['num_blocks']

        self.embedding = nn.Embedding(self.vocab_size, params['d_model'])
        self.pos_embedding = PositionalEncoding(params['d_model'], 0.0)

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                params['n_heads'], params['d_model'], params['d_ff'],
                slf_attn_dropout=0.0, ffn_dropout=0.0,
                residual_dropout=params['residual_dropout'],
                normalize_before=False, concat_after=False, activation='glu') for _ in range(self.num_blocks)
        ])

        if self.normalize_before:
            self.after_norm = nn.LayerNorm(params['d_model'])

        self.output_project = nn.Linear(params['d_model'], self.vocab_size)

        if params['share_embedding']:
            self.output_project.weight = self.embedding.weight
            print('Share the weight of embedding to the output project layer!')

        self.crit = LabelSmoothingLoss(size=self.vocab_size, smoothing=self.smoothing, padding_idx=PAD)

    def forward(self, inputs, targets):

        dec_output = self.embedding(inputs['inputs'])
        dec_mask = get_seq_mask(inputs['inputs'])
        dec_output = self.pos_embedding(dec_output)

        for _, block in enumerate(self.blocks):
            dec_output, _ = block(dec_output, dec_mask)

        if self.normalize_before:
            dec_output = self.after_norm(dec_output)

        logits = self.output_project(dec_output)
        loss = self.crit(logits, targets['targets'])

        return loss, None

    def predict(self, targets, last_frame=True):

        dec_output = self.embedding(targets)
        dec_output = self.pos_embedding(dec_output)

        dec_mask = get_seq_mask(targets)

        for _, block in enumerate(self.blocks):
            dec_output, dec_mask = block(dec_output, dec_mask)

        if self.normalize_before:
            dec_output = self.after_norm(dec_output)

        logits = self.output_project(dec_output)

        if last_frame:
            log_probs = F.log_softmax(logits[:, -1, :].unsqueeze(1), dim=-1)
        else:
            log_probs = F.log_softmax(logits, dim=-1)

        return log_probs

    def save_checkpoint(self, params, name):
        checkpoint = {
            'params': params,
            'model': self.state_dict()
            }

        torch.save(checkpoint, name)
