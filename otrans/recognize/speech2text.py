import torch
from otrans.data import EOS, BOS
from otrans.recognize.base import Recognizer
from packaging import version

class SpeechToTextRecognizer(Recognizer):
    def __init__(self, model, lm=None, lm_weight=0.1, ctc_weight=0.0, beam_width=5, nbest=1,
                 max_len=50, idx2unit=None, penalty=0, lamda=5, ngpu=1, apply_cache=False):
        super(SpeechToTextRecognizer, self).__init__(model, idx2unit, lm, lm_weight, ngpu)

        self.beam_width = beam_width
        self.max_len = max_len
        self.nbest = nbest

        self.penalty = penalty
        self.lamda = lamda

        self.ctc_weight = ctc_weight
        self.lm_weight = lm_weight

        self.attn_weights = {}
        self.apply_cache = False 

    def encode(self, inputs, inputs_mask, cache=None):
        new_cache = {}
        inputs, inputs_mask, fe_cache = self.model.frontend.inference(inputs, inputs_mask, cache['frontend'] if cache is not None else None)
        new_cache['frontend'] = fe_cache

        memory, memory_mask, enc_cache, enc_attn_weights = self.model.encoder.inference(inputs, inputs_mask, cache['encoder'] if cache is not None else None)
        new_cache['encoder'] = enc_cache

        return memory, memory_mask, new_cache, enc_attn_weights

    def decode(self, preds, memory, memory_mask, cache=None):
        log_probs, dec_cache, dec_attn_weights = self.model.decoder.inference(preds, memory, memory_mask, cache)
        return log_probs, dec_cache, dec_attn_weights

    def recognize(self, inputs, inputs_mask):

        cache = {'fronend': None, 'encoder': None, 'decoder': None, 'lm': None}

        self.attn_weights = {}
        memory, memory_mask, _, enc_attn_weights = self.encode(inputs, inputs_mask)
        
        self.attn_weights['encoder'] = enc_attn_weights
        self.attn_weights['decoder'] = []

        b, t, v = memory.size()

        beam_memory = memory.unsqueeze(1).repeat([1, self.beam_width, 1, 1]).view(b * self.beam_width, t, v)
        beam_memory_mask = memory_mask.unsqueeze(1).repeat([1, self.beam_width, 1]).view(b * self.beam_width, t)

        preds = torch.ones([b * self.beam_width, 1], dtype=torch.long, device=memory.device) * BOS

        scores = torch.FloatTensor([0.0] + [-float('inf')] * (self.beam_width - 1))
        scores = scores.to(memory.device).repeat([b]).unsqueeze(1)
        ending_flag = torch.zeros_like(scores, dtype=torch.bool)


        with torch.no_grad():
            for _ in range(1, self.max_len+1):
                preds, cache, scores, ending_flag = self.decode_step(
                    preds, beam_memory, beam_memory_mask, cache, scores, ending_flag)

                # whether stop or not
                if ending_flag.sum() == b * self.beam_width:
                    break

            scores = scores.view(b, self.beam_width)
            preds = preds.view(b, self.beam_width, -1)

            lengths = torch.sum(torch.ne(preds, EOS).float(), dim=-1)

            # length penalty
            if self.penalty:
                lp = torch.pow((self.lamda + lengths) /
                               (self.lamda + 1), self.penalty)
                scores /= lp

            sorted_scores, offset_indices = torch.sort(scores, dim=-1, descending=True)

            base_indices = torch.arange(b, dtype=torch.long, device=offset_indices.device) * self.beam_width
            base_indices = base_indices.unsqueeze(1).repeat([1, self.beam_width]).view(-1)
            preds = preds.view(b * self.beam_width, -1)
            indices = offset_indices.view(-1) + base_indices

            # remove BOS
            sorted_preds = preds[indices].view(b, self.beam_width, -1)
            nbest_preds = sorted_preds[:, :min(self.beam_width, self.nbest), 1:]
            nbest_scores = sorted_scores[:, :min(self.beam_width, self.nbest)]

        return self.nbest_translate(nbest_preds), nbest_scores

    def decode_step(self, preds, memory, memory_mask, cache, scores, flag):
        """ decode an utterance in a stepwise way"""

        batch_size = int(scores.size(0) / self.beam_width)

        batch_log_probs, dec_cache, dec_attn_weights = self.decode(preds, memory, memory_mask, cache['decoder'])

        if self.lm is not None:
            batch_lm_log_probs, lm_hidden = self.lm_decode(preds, cache['lm'])
            batch_lm_log_probs = batch_lm_log_probs.squeeze(1)
            batch_log_probs = batch_log_probs + self.lm_weight * batch_lm_log_probs
        else:
            lm_hidden = None

        if batch_log_probs.dim() == 3:
            batch_log_probs = batch_log_probs.squeeze(1)

        last_k_scores, last_k_preds = batch_log_probs.topk(self.beam_width)

        last_k_scores = mask_finished_scores(last_k_scores, flag)
        last_k_preds = mask_finished_preds(last_k_preds, flag)

        # update scores
        scores = scores + last_k_scores
        scores = scores.view(batch_size, self.beam_width * self.beam_width)

        # pruning
        scores, offset_k_indices = torch.topk(scores, k=self.beam_width)
        scores = scores.view(-1, 1)

        device = scores.device
        base_k_indices = torch.arange(batch_size, device=device).view(-1, 1).repeat([1, self.beam_width])
        base_k_indices *= self.beam_width ** 2
        best_k_indices = base_k_indices.view(-1) + offset_k_indices.view(-1)

        # update predictions
        best_k_preds = torch.index_select(
            last_k_preds.view(-1), dim=-1, index=best_k_indices)
        if version.parse(torch.__version__) < version.parse('1.6.0'):
            preds_index = best_k_indices.div(self.beam_width)
        else:
            preds_index = best_k_indices.floor_divide(self.beam_width)
        preds_symbol = torch.index_select(
            preds, dim=0, index=preds_index)
        preds_symbol = torch.cat(
            (preds_symbol, best_k_preds.view(-1, 1)), dim=1)

        # dec_hidden = reselect_hidden_list(dec_hidden, self.beam_width, best_k_indices)
        # lm_hidden = reselect_hidden_list(lm_hidden, self.beam_width, best_k_indices)

        # finished or not
        end_flag = torch.eq(preds_symbol[:, -1], EOS).view(-1, 1)

        # hidden = {
        #     'decoder': dec_hidden,
        #     'lm': lm_hidden
        # }

        return preds_symbol, cache, scores, end_flag


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
        unfinished = torch.cat(
            (zero_mask, flag.repeat([1, beam_width - 1])), dim=1)
        finished = torch.cat(
            (flag.bool(), zero_mask.repeat([1, beam_width - 1])), dim=1)
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


def reselect_hidden(tensor, beam_width, indices):
    n_layers, batch_size, hidden_size = tensor.size()
    tensor = tensor.transpose(0, 1).unsqueeze(1).repeat([1, beam_width, 1, 1])
    tensor = tensor.reshape(batch_size * beam_width, n_layers, hidden_size)
    new_tensor = torch.index_select(tensor, dim=0, index=indices)
    new_tensor = new_tensor.transpose(0, 1).contiguous()
    return new_tensor


def reselect_hidden_list(tensor_list, beam_width, indices):

    if tensor_list is None:
        return None

    new_tensor_list = []
    for tensor in tensor_list:
        if isinstance(tensor, tuple):
            h = reselect_hidden(tensor[0], beam_width, indices)
            c = reselect_hidden(tensor[1], beam_width, indices)
            new_tensor_list.append((h, c))
        else:
            new_tensor_list.append(reselect_hidden(tensor, beam_width, indices))
    
    return new_tensor_list
