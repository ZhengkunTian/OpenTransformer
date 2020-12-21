from otrans.recognize.ctc import CTCRecognizer
from otrans.recognize.speech2text import SpeechToTextRecognizer


def build_recognizer(model_type, model, lm, args, idx2unit):
    if model_type == 'speech2text':
        return SpeechToTextRecognizer(
            model=model, lm=lm, lm_weight=args.lm_weight,
            ctc_weight=args.ctc_weight, beam_width=args.beam_width, nbest=args.nbest, max_len=args.max_len,
            idx2unit=idx2unit, penalty=args.penalty, lamda=args.lamda, ngpu=args.ngpu)
    elif model_type == 'ctc':
        return CTCRecognizer(
            model=model, lm=lm, lm_weight=args.lm_weight, ngram_lm=args.ngram_lm, beam_width=args.beam_width,
            idx2unit=idx2unit, ngpu=args.ngpu, mode=args.mode, alpha=args.alpha, beta=args.beta)
    else:
        raise NotImplementedError
