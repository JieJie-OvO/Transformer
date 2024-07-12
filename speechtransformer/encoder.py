import torch
import torch.nn as nn
from .backbone.pos import PositionalEncoding
from .backbone.feedforward import PositionwiseFeedForward, FeedForwardModule
from .backbone.attention import MultiHeadedSelfAttention
from .backbone.convblock import ConformerConvModule

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

    def forward(self, x, mask = None):
        residual = x
        if self.pre_norm:
            x = self.norm1(x)
        x, attn = self.attn(x, mask)
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
    

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=2048, n_layers=12, 
                 attn_drop=0.0, ffn_drop=0.0, residual_drop=0.1, pre_norm=False):
        
        super(TransformerEncoder, self).__init__()

        self.pre_norm = pre_norm

        self.pos_emb = PositionalEncoding(d_model)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(EncoderLayer(n_heads, d_model, d_ff, attn_drop, ffn_drop, residual_drop, pre_norm))

        if pre_norm:
            self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs, mask):
    
        enc_output, pos = self.pos_emb(inputs)

        attns = []
        for layer in self.layers:
            enc_output, attn = layer(enc_output, mask.unsqueeze(1))
            attns.append(attn)

        if self.pre_norm:
            enc_output = self.norm(enc_output)

        return enc_output, mask, {"attns":attns, "pos":pos}