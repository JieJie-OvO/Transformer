import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.pos import PositionalEncoding
from .backbone.feedforward import PositionwiseFeedForward, FeedForwardModule
from .backbone.attention import MultiHeadedCrossAttention, MultiHeadedSelfAttention
from .backbone.utils import get_transformer_decoder_mask
from .backbone.vab import PAD

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, enc_dim, slf_attn_drop=0.0, src_attn_drop=0.0, 
                 ffn_drop=0.0, residual_dropout=0.1, pre_norm=False):
        super(DecoderLayer, self).__init__()

        self.attn1 = MultiHeadedSelfAttention(n_heads, d_model, slf_attn_drop)
        self.attn2 = MultiHeadedCrossAttention(n_heads, d_model, enc_dim, src_attn_drop)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, ffn_drop)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(residual_dropout)
        self.dropout2 = nn.Dropout(residual_dropout)
        self.dropout3 = nn.Dropout(residual_dropout)

        self.pre_norm = pre_norm


    def forward(self, x, enc_out, self_mask=None, cross_mask = None):

        residual = x
        if self.pre_norm:
            x = self.norm1(x)
        
        x, attn1 = self.attn1(x, self_mask)

        x = residual + self.dropout1(x)

        if not self.pre_norm:
            x = self.norm1(x)

        residual = x
        if self.pre_norm:
            x = self.norm2(x)

        x, attn2 = self.attn2(x, enc_out, cross_mask)
        
        x = residual + self.dropout2(x)

        if not self.pre_norm:
            x = self.norm2(x)

        residual = x
        if self.pre_norm:
            x = self.norm3(x)
        
        x = self.feed_forward(x)
        x = residual + self.dropout3(x)

        if not self.pre_norm:
            x = self.norm3(x)

        return x, attn1, attn2


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size=4233, d_model=256, n_heads=4, d_ff=2048, enc_dim=256, n_layers=6, 
                 slf_attn_drop=0.0, src_attn_drop=0.0, ffn_drop=0.0, residual_drop=0.1, pre_norm=False):
        
        super(TransformerDecoder, self).__init__()

        self.pre_norm = pre_norm

        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_emb = PositionalEncoding(d_model)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(DecoderLayer(n_heads, d_model, d_ff, enc_dim, slf_attn_drop, 
                                            src_attn_drop, ffn_drop, residual_drop, pre_norm=pre_norm))

        if self.pre_norm:
            self.after_norm = nn.LayerNorm(d_model)

        self.output_layer = nn.Linear(d_model, vocab_size)

        self.output_layer.weight = self.embedding.weight

    def forward(self, input, enc_out, enc_mask):

        x = self.embedding(input)
        x, pos = self.pos_emb(x)

        self_mask = get_transformer_decoder_mask(input)

        attn1s = []
        attn2s = []

        for layer in self.layers:
            x, attn1, attn2 = layer(x, enc_out, self_mask, enc_mask.unsqueeze(1))
            attn1s.append(attn1)
            attn2s.append(attn2)

        if self.pre_norm:
            x = self.after_norm(x)

        logits = self.output_layer(x)

        return logits, {"attn1":attn1, "attn2":attn2}

    def inference(self, preds, memory, memory_mask=None):

        logits, attn = self.forward(preds,  memory, memory_mask)

        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

        return log_probs, attn



