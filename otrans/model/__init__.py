from .transformer import Transformer


def End2EndModel(params):

    if params['type'] == 'transformer':
        return Transformer(params)
    else:
        raise NotImplementedError