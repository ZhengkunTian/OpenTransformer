"""
@Author: Zhengkun Tian
@Email: zhengkun.tian@outlook.com
@Date: 2020-04-23 15:14:28
@LastEditTime: 2020-04-23 15:16:49
@FilePath: \OpenTransformer\otrans\recognizer.py
"""
import torch
from otrans.data import BOS, EOS, normalization


def mask_finished_scores(score, flag):
    """
    If a sequence is finished, we only allow one alive branch. This function aims to give one branch a zero score
    and the rest -inf score.
    Args:
        score: A real value array with shape [batch_size * beam_size, beam_size].
        flag: A bool array with shape [batch_size * beam_size, 1].
    Returns:
        A real value array with shape [batch_size * beam_size, beam_size].
    """
    beam_width = score.size(-1)
    zero_mask = torch.zeros_like(flag, dtype=torch.bool)
    if beam_width > 1:
        unfinished = torch.cat((zero_mask, flag.repeat([1, beam_width - 1])), dim=1)
        finished = torch.cat((flag.bool(), zero_mask.repeat([1, beam_width - 1])), dim=1)
    else:
        unfinished = zero_mask
        finished = flag.bool()
    score.masked_fill_(unfinished, -float('inf'))
    score.masked_fill_(finished, 0)
    return score


def mask_finished_preds(pred, flag):
    """
    If a sequence is finished, all of its branch should be </S> (3).
    Args:
        pred: A int array with shape [batch_size * beam_size, beam_size].
        flag: A bool array with shape [batch_size * beam_size, 1].
    Returns:
        A int array with shape [batch_size * beam_size].
    """
    beam_width = pred.size(-1)
    finished = flag.repeat([1, beam_width])
    return pred.masked_fill_(finished.bool(), EOS)


class TransformerRecognizer(object):
    def __init__(self, model, lm=None, lm_weight=0.1, beam_width=5, max_len=50, unit2char=None,
                 penalty=0, lamda=5, ngpu=1):

        self.model = model
        self.lm = lm
        self.lm_weight = lm_weight
        self.beam_width = beam_width
        self.max_len = max_len
        self.unit2char = unit2char

        self.penalty = penalty
        self.lamda = lamda

        self.model.eval()
        if self.lm is not None:
            self.lm.eval()
        self.ngpu = ngpu

    def recognize(self, inputs, inputs_length):

        enc_states, enc_masks = self.encode(inputs, inputs_length)

        b, t, v = enc_states.size()

        beam_enc_states = enc_states.unsqueeze(1).repeat([1, self.beam_width, 1, 1]).view(b * self.beam_width, t, v)
        beam_enc_mask = enc_masks.unsqueeze(1).repeat([1, self.beam_width, 1, 1]).view(b * self.beam_width, 1, t)

        preds = torch.ones([b * self.beam_width, 1], dtype=torch.long, device=enc_states.device) * BOS

        scores = torch.FloatTensor([0.0] + [-float('inf')] * (self.beam_width - 1))
        scores = scores.to(enc_states.device).repeat([b]).unsqueeze(1)
        ending_flag = torch.zeros_like(scores, dtype=torch.bool)

        with torch.no_grad():

            for step in range(1, self.max_len+1):

                preds, scores, ending_flag = self.decode_step(preds, beam_enc_states, beam_enc_mask,
                                                              scores, ending_flag)

                # whether stop or not
                if ending_flag.sum() == b * self.beam_width:
                    break

            scores = scores.view(b, self.beam_width)
            preds = preds.view(b, self.beam_width, -1)

            lengths = torch.sum(torch.ne(preds, EOS).float(), dim=-1)

            # length penalty
            if self.penalty:
                lp = torch.pow((self.lamda + lengths) / (self.lamda + 1), self.penalty)
                scores /= lp

            max_indices = torch.argmax(scores, dim=-1).long()
            max_indices += torch.arange(b, dtype=torch.long, device=max_indices.get_device()) * self.beam_width
            preds = preds.view(b * self.beam_width, -1)

            best_preds = torch.index_select(preds, dim=0, index=max_indices)
            # remove BOS
            best_preds = best_preds[:, 1:]

            results = []
            for pred in best_preds:
                preds = []
                for i in pred:
                    if int(i) == EOS:
                        break
                    preds.append(self.unit2char[int(i)])
                results.append(' '.join(preds))

        return results

    def encode(self, inputs, inputs_length):
        enc_states, enc_mask = self.model.encoder(inputs, inputs_length)
        return enc_states, enc_mask

    def decode_step(self, preds, enc_state, enc_mask, scores, flag):
        """ decode an utterance in a stepwise way"""

        batch_size = int(scores.size(0) / self.beam_width)

        batch_log_probs = self.model.decoder.recognize(preds, enc_state, enc_mask).detach()

        if self.lm is not None:
            batch_lm_log_probs = self.lm.predict(preds)
            batch_lm_log_probs = batch_lm_log_probs.squeeze(1)
            batch_log_probs = batch_log_probs + self.lm_weight * batch_lm_log_probs

        last_k_scores, last_k_preds = batch_log_probs.topk(self.beam_width)

        last_k_scores = mask_finished_scores(last_k_scores, flag)
        last_k_preds = mask_finished_preds(last_k_preds, flag)

        # update scores
        scores = scores + last_k_scores
        scores = scores.view(batch_size, self.beam_width * self.beam_width)

        # pruning
        scores, offset_k_indices = torch.topk(scores, k=self.beam_width)
        scores = scores.view(-1, 1)

        device = scores.get_device()
        base_k_indices = torch.arange(batch_size, device=device).view(-1, 1).repeat([1, self.beam_width])
        base_k_indices *= self.beam_width ** 2
        best_k_indices = base_k_indices.view(-1) + offset_k_indices.view(-1)

        # update predictions
        best_k_preds = torch.index_select(last_k_preds.view(-1), dim=-1, index=best_k_indices)
        preds_symbol = torch.index_select(preds, dim=0, index=best_k_indices.div(self.beam_width))
        preds_symbol = torch.cat((preds_symbol, best_k_preds.view(-1, 1)), dim=1)

        # finished or not
        end_flag = torch.eq(preds_symbol[:, -1], EOS).view(-1, 1)

        return preds_symbol, scores, end_flag
