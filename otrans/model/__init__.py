from .ctc import CTCModel
from .speech2text import SpeechToText
from .lm import RecurrentLanguageModel, TransformerLanguageModel


End2EndModel = {
    'ctc': CTCModel,
    'speech2text': SpeechToText
}

LanguageModel = {
    'rnn_lm': RecurrentLanguageModel,
    'transformer_lm': TransformerLanguageModel
}
