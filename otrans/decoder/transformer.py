# File   : transformer.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from otrans.module.pos import PositionalEncoding
from otrans.module.ffn import PositionwiseFeedForward
from otrans.module.attention import MultiHeadedSelfAttention, MultiHeadedCrossAttention, MultiHeadedSelfAttentionWithRelPos
from otrans.data import PAD
from otrans.decoder.utils import get_transformer_decoder_mask

logger = logging.getLogger(__name__)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, memory_dim, slf_attn_dropout=0.0, src_attn_dropout=0.0, ffn_dropout=0.0, residual_dropout=0.1,
                 normalize_before=False, concat_after=False, relative_positional=False, activation='relu'):
        super(TransformerDecoderLayer, self).__init__()

        self.relative_positional = relative_positional

        if self.relative_positional:
            self.slf_attn = MultiHeadedSelfAttentionWithRelPos(n_heads, d_model, slf_attn_dropout)
        else:
            self.slf_attn = MultiHeadedSelfAttention(n_heads, d_model, slf_attn_dropout)
        self.src_attn = MultiHeadedCrossAttention(n_heads, d_model, memory_dim, src_attn_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, ffn_dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(residual_dropout)
        self.dropout2 = nn.Dropout(residual_dropout)
        self.dropout3 = nn.Dropout(residual_dropout)

        self.normalize_before = normalize_before
        self.concat_after = concat_after

        if self.concat_after:
            self.concat_linear1 = nn.Linear(d_model * 2, d_model)
            self.concat_linear2 = nn.Linear(d_model * 2, d_model)

    def forward(self, tgt, tgt_mask, memory, memory_mask, pos):
        """Compute decoded features

        :param torch.Tensor tgt: decoded previous target features (batch, max_time_out, size)
        :param torch.Tensor tgt_mask: mask for x (batch, max_time_out)
        :param torch.Tensor memory: encoded source features (batch, max_time_in, size)
        :param torch.Tensor memory_mask: mask for memory (batch, max_time_in)
        """

        if self.normalize_before:
            tgt = self.norm1(tgt)
        residual = tgt

        if self.relative_positional:
            slf_attn_out, slf_attn_weights = self.slf_attn(tgt, tgt_mask, pos)
        else:
            slf_attn_out, slf_attn_weights = self.slf_attn(tgt, tgt_mask)

        if self.concat_after:
            x = residual + self.concat_linear1(torch.cat((tgt, slf_attn_out), dim=-1))
        else:
            x = residual + self.dropout1(slf_attn_out)
        if not self.normalize_before:
            x = self.norm1(x)

        if self.normalize_before:
            x = self.norm2(x)
        residual = x
        src_attn_out, src_attn_weights = self.src_attn(x, memory, memory_mask)
        if self.concat_after:
            x = residual + self.concat_linear2(torch.cat((x, src_attn_out), dim=-1))
        else:
            x = residual + self.dropout2(src_attn_out)
        if not self.normalize_before:
            x = self.norm2(x)

        if self.normalize_before:
            x = self.norm3(x)
        residual = x
        x = residual + self.dropout3(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        return x, {'slf_attn_weights': slf_attn_weights, 'src_attn_weights': src_attn_weights}

    def inference(self, x, xmask, memory, memory_mask=None, pos=None, cache={'slf': None, 'src': None}):

        if self.normalize_before:
            x = self.norm1(x)
        residual = x
        if self.relative_positional:
            slf_attn_out, slf_attn_weight, slf_cache = self.slf_attn.inference(x, xmask, pos, cache=['slf'])
        else:
            slf_attn_out, slf_attn_weight, slf_cache = self.slf_attn.inference(x, xmask, cache=['slf'])
        if self.concat_after:
            x = residual + self.concat_linear1(torch.cat((x, slf_attn_out), dim=-1))
        else:
            x = residual + self.dropout1(slf_attn_out)
        if not self.normalize_before:
            x = self.norm1(x)

        if self.normalize_before:
            x = self.norm2(x)
        residual = x
        src_attn_out, src_attn_weight, src_cache = self.src_attn.inference(x, memory, memory_mask, cache['src'])
        if self.concat_after:
            x = residual + self.concat_linear2(torch.cat((x, src_attn_out), dim=-1))
        else:
            x = residual + self.dropout2(src_attn_out)
        if not self.normalize_before:
            x = self.norm2(x)

        if self.normalize_before:
            x = self.norm3(x)
        residual = x
        x = residual + self.dropout3(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        return x, {'slf_attn_weight': slf_attn_weight, 'src_attn_weight': src_attn_weight}, {'slf': slf_cache, 'src': src_cache}


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, d_ff=2048, memory_dim=256, n_blocks=6, pos_dropout=0.0, slf_attn_dropout=0.0, src_attn_dropout=0.0, ffn_dropout=0.0,
                 residual_dropout=0.1, activation='relu', normalize_before=True, concat_after=False, share_embedding=False):
        super(TransformerDecoder, self).__init__()

        self.decoder_type = 'transformer'
        self.normalize_before = normalize_before
        self.relative_positional = False

        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_emb = PositionalEncoding(d_model, pos_dropout)

        self.blocks = nn.ModuleList([
            TransformerDecoderLayer(
                n_heads, d_model, d_ff, memory_dim, slf_attn_dropout, src_attn_dropout,
                ffn_dropout, residual_dropout, normalize_before=normalize_before, concat_after=concat_after,
                relative_positional=False, activation=activation) for _ in range(n_blocks)
        ])

        if self.normalize_before:
            self.after_norm = nn.LayerNorm(d_model)

        self.output_layer = nn.Linear(d_model, vocab_size)

        if share_embedding:
            assert self.embedding.weight.size() == self.output_layer.weight.size()
            self.output_layer.weight = self.embedding.weight
            logger.info('Tie the weights between the embedding and output layer.')

    def forward(self, targets, memory, memory_mask):

        dec_output = self.embedding(targets)
        if self.relative_positional:
            # [1, 2T - 1]
            position = torch.arange(-(dec_output.size(1)-1), dec_output.size(1), device=dec_output.device).reshape(1, -1)
            pos = self.pos_emb._embedding_from_positions(position)
        else:  
            dec_output, pos = self.pos_emb(dec_output)

        dec_mask = get_transformer_decoder_mask(targets)

        attn_weights = {}
        for i, block in enumerate(self.blocks):
            dec_output, attn_weight = block(dec_output, dec_mask, memory, memory_mask.unsqueeze(1), pos)
            attn_weights['dec_block_%d' % i] = attn_weight

        if self.normalize_before:
            dec_output = self.after_norm(dec_output)

        logits = self.output_layer(dec_output)

        return logits, attn_weights

    def inference(self, preds, memory, memory_mask=None, cache=None):

        assert preds.dim() == 2
        # dec_output = self.embedding(preds)
        # dec_output, pos = self.pos_encoding.inference(dec_output)
        # mask = get_transformer_decoder_mask(preds)

        # new_caches = []
        # attn_weights = {}
        # for i, block in enumerate(self.blocks):
        #     block_cache = cache[i] if cache is not None else {'slf': None, 'src': None}
        #     dec_output, attn_weight, block_cache = block.inference(dec_output, mask, memory, memory_mask.unsqueeze(1), pos, cache=block_cache)
        #     attn_weights['dec_block_%d' % i] = attn_weight
        #     new_caches.append(block_cache)

        # if self.normalize_before:
        #     dec_output = self.after_norm(dec_output)

        # logits = self.output_layer(dec_output) # logits [batch_size, 1, model_size]
        logits, attn_weights = self.forward(preds,  memory, memory_mask)

        log_probs = F.log_softmax(logits[:, -1, :], dim=-1) # logits [batch_size, 1, model_size]

        return log_probs, cache, attn_weights


