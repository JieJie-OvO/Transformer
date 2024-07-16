import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .attention import MultiHeadedSelfAttention
from .feedforward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, slf_attn_dropout=0.0, ffn_dropout=0.0, residual_dropout=0.1, pre_norm=False):
        super(EncoderLayer, self).__init__()

        self.attn = MultiHeadedSelfAttention(n_heads, d_model, slf_attn_dropout)
        
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, ffn_dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(residual_dropout)
        self.dropout2 = nn.Dropout(residual_dropout)

        self.pre_norm = pre_norm

    def forward(self, x, pad_mask = None, mask = None):
        residual = x
        if self.pre_norm:
            x = self.norm1(x)
        x, attn = self.attn(x, pad_mask, mask)
        x = residual + self.dropout1(x)
        if not self.pre_norm:
            x = self.norm1(x)

        residual = x
        if self.pre_norm:
            x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)

        return x, attn
    
