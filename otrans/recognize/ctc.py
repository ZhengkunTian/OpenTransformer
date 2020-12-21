import torch
import logging
from otrans.data import PAD, BLK, EOS
from otrans.recognize.base import Recognizer


class CTCRecognizer(Recognizer):

    def __init__(self, model, lm=None, lm_weight=0.1, ngram_lm=None, beam_width=5,
                 idx2unit=None, ngpu=1, mode='greedy', alpha=0.1, beta=0.0):
        super().__init__(model, idx2unit, lm, lm_weight, ngpu)
        
        self.beam_width = beam_width
        self.mode = mode

        # self.ctcdecode = CTCBeamSearch(BLK, self.beam_width, self.beam_width, idx2unit=None, ngram_lm=None, lm_weight=0.0, keep_n_tokens=40)

        if self.mode == 'beam':
            import ctcdecode_edited as ctcdecode
            # import ctcdecode
            vocab_list = [self.idx2unit[i] for i in range(len(idx2unit))]
            self.ctcdecoder = ctcdecode.CTCBeamDecoder(
                vocab_list, beam_width=self.beam_width,
                blank_id=BLK, model_path=ngram_lm, alpha=alpha, beta=beta,
                log_probs_input=True, num_processes=10)

    def recognize(self, inputs, inputs_length):

        if self.mode == 'greedy':
            results = self.recognize_greedy(inputs, inputs_length)
        elif self.mode == 'beam':
            results = self.recognize_beam(inputs, inputs_length)
        else:
            raise ValueError

        return self.translate(results)
        
    def recognize_greedy(self, inputs, inputs_length):

        log_probs, length = self.model.inference(inputs, inputs_length)

        _, preds = log_probs.topk(self.beam_width, dim=-1)

        results = []
        for b in range(log_probs.size(0)):
            pred = []
            last_k = PAD
            for i in range(int(length[b])):
                k = int(preds[b][i][0])
                if k == last_k or k == PAD:
                    last_k = k
                    continue
                else:
                    last_k = k
                    pred.append(k)

            results.append(pred)
        return results

    def recognize_beam(self, inputs, inputs_length):

        log_probs, length = self.model.inference(inputs, inputs_length)

        beam_results, beam_scores, _, out_seq_len = self.ctcdecoder.decode(log_probs.cpu(), seq_lens=length.cpu())

        best_results = beam_results[:, 0]
        batch_length = out_seq_len[:, 0]
        # print(beam_scores)

        # print(best_results)

        results = []
        for b in range(log_probs.size(0)):
            length = int(batch_length[b])
            tokens = [int(i) for i in best_results[b, :length]]
            results.append(tokens)

        return results


