# File   : __init__.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com


from otrans.encoder.transformer import TransformerEncoder
from otrans.encoder.conformer import ConformerEncoder


BuildEncoder = {
    'transformer': TransformerEncoder,
    'conformer': ConformerEncoder,
}
