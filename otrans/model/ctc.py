import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from otrans.frontend import BuildFrontEnd
from otrans.model.base import BaseModel
from otrans.encoder import BuildEncoder
from otrans.data import BLK

logger = logging.getLogger(__name__)

class CTCAssistor(nn.Module):
    def __init__(self, hidden_size, vocab_size, blank=BLK, lookahead_steps=-1):
        super(CTCAssistor, self).__init__()

        self.lookahead_steps = lookahead_steps
        if self.lookahead_steps > 0:
            self.apply_look_ahead = True
            self.lookahead_conv = nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=self.lookahead_steps + 1,
                    padding=0, stride=1, bias=False,
                    groups=hidden_size)
            logger.info('Apply Lookahead Step in CTCAssistor And Set it to %d' % lookahead_steps)       
        else:
            self.apply_look_ahead = False 

        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.ctc_crit = nn.CTCLoss(blank=blank, zero_infinity=True)

    def forward(self, memory, memory_length=None, targets=None, tgt_length=None, return_logits=False):

        if self.apply_look_ahead:
            memory = F.pad(memory, pad=(0, 0, 0, self.lookahead_steps), value=0.0)
            memory = memory.transpose(1, 2)
            memory = self.lookahead_conv(memory)
            memory = memory.transpose(1, 2)

        logits = self.compute_logits(memory)
        if return_logits:
            return logits
        else:
            loss = self.compute_loss(logits, memory_length, targets, tgt_length)
            return loss

    def compute_logits(self, enc_states):
        return self.output_layer(enc_states)

    def compute_loss(self, logits, enc_length, targets, targets_length):
        log_probs = F.log_softmax(logits, dim=-1)
        loss = self.ctc_crit(log_probs.transpose(0, 1), targets, enc_length, targets_length)
        return loss

    def inference(self, memory, memory_mask):

        if self.apply_look_ahead:
            memory = F.pad(memory, pad=(0, 0, 0, self.lookahead_steps), value=0.0)
            memory = memory.transpose(1, 2)
            memory = self.lookahead_conv(memory)
            memory = memory.transpose(1, 2)

        logits = self.output_layer(memory)
        memory_length = torch.sum(memory_mask.squeeze(1), dim=-1)

        return F.log_softmax(logits, dim=-1), memory_length        


class CTCModel(BaseModel):
    def __init__(self, params):
        super(CTCModel, self).__init__()

        self.frontend = BuildFrontEnd[params['frontend_type']](**params['frontend'])
        logger.info('Build a %s frontend!' % params['frontend_type'])
        self.encoder = BuildEncoder[params['encoder_type']](**params['encoder'])
        logger.info('Build a %s encoder!' % params['encoder_type'])

        self.assistor = CTCAssistor(
            hidden_size=params['encoder_output_size'],
            vocab_size=params['vocab_size'],
            lookahead_steps=params['lookahead_steps'] if 'lookahead_steps' in params else -1)

    def forward(self, inputs, targets):

        enc_inputs = inputs['inputs']
        enc_mask = inputs['mask']

        truth = targets['targets']
        truth_length = targets['targets_length']

        enc_inputs, enc_mask = self.frontend(enc_inputs, enc_mask)
        memory, memory_mask, _ = self.encoder(enc_inputs, enc_mask)

        memory_length = torch.sum(memory_mask, dim=-1)
        loss = self.assistor(memory, memory_length, truth[:, 1:-1], truth_length.add(-1))
        return loss, None

    def inference(self, inputs, inputs_length):
        memory, memory_mask = self.encoder(inputs, inputs_length)
        logits = self.assistor(memory, return_logits=True)
        memory_length = torch.sum(memory_mask.squeeze(1), dim=-1)

        return F.log_softmax(logits, dim=-1), memory_length        

    def ts_forward(self, inputs, inputs_length, targets, targets_length, return_loss=True):
        memory, memory_mask = self.encoder(inputs, inputs_length)
        logits = self.assistor.output_layer(memory)

        if return_loss:
            memory_length = torch.sum(memory_mask.squeeze(1), dim=-1)
            targets_out = targets[:, 1:].clone()
            loss = self.assistor.compute_loss(logits, memory_length, targets_out, targets_length)
            return loss, logits, memory_mask

        return logits, memory_mask

    def recognize(self, inputs, inputs_length):
        memory, memory_mask = self.encoder(inputs, inputs_length)
        logits = self.assistor(memory, return_logits=True)
        memory_length = torch.sum(memory_mask.squeeze(1), dim=-1)
        return F.log_softmax(logits, dim=-1), memory_length

    def save_checkpoint(self, params, name):
        checkpoint = {
            'params': params,
            'frontend': self.frontend.state_dict(),
            'encoder': self.encoder.state_dict(),
            'ctc': self.assistor.state_dict()
            }

        torch.save(checkpoint, name)

    def set_epoch(self, epoch):
        pass
