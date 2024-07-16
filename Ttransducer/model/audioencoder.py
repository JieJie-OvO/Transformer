import torch
import torch.nn as nn
from .backbone.layer import EncoderLayer
from .backbone.pos import PositionalEncoding
from .backbone.mask_strategy import *

class AudioEncoder(nn.Module):
    def __init__(self, fbank=80, d_model=256, n_heads=4, d_ff=2048, n_layers=12, 
                 attn_drop=0.0, ffn_drop=0.0, residual_drop=0.1, pre_norm=False,
                 chunk_size=10):
        super(AudioEncoder, self).__init__()

        self.pre_norm = pre_norm

        self.linear = nn.Linear(fbank, d_model)

        self.pos_emb = PositionalEncoding(d_model)

        self.layers = nn.ModuleList()

        for i in range(n_layers):
            self.layers.append(EncoderLayer(n_heads, d_model, d_ff, attn_drop, ffn_drop, residual_drop, pre_norm))

        if pre_norm:
            self.norm = nn.LayerNorm(d_model)
        
        self.chunksize = chunk_size

    def forward(self, inputs, pad_mask=None):
        enc_output = self.linear(inputs)
        enc_output, pos = self.pos_emb(enc_output)

        mask = get_chunk_lookahead_mask(inputs, self.chunksize)

        attns = []
        for layer in self.layers:
            enc_output, attn = layer(enc_output, pad_mask.unsqueeze(1), mask)
            attns.append(attn)

        if self.pre_norm:
            enc_output = self.norm(enc_output)

        return enc_output, pad_mask, {"attns":attns, "pos":pos}