import torch
import torch.nn as nn
import numpy as np
from .backbone.layer import EncoderLayer
from .backbone.pos import PositionalEncoding
from .backbone.mask_strategy import *

class LabelEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, d_ff=2048, n_layers=12, 
                 attn_drop=0.0, ffn_drop=0.0, residual_drop=0.1, pre_norm=False):
        super(LabelEncoder, self).__init__()

        self.pre_norm = pre_norm

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(EncoderLayer(n_heads, d_model, d_ff, attn_drop, ffn_drop, residual_drop, pre_norm))

        if pre_norm:
            self.norm = nn.LayerNorm(d_model)
        
    def forward(self, input, pad_mask=None):
        x = self.embedding(input)
        enc_output, pos = self.pos_emb(x)

        mask = get_tril_mask(input)

        attns = []
        for layer in self.layers:
            enc_output, attn = layer(enc_output, pad_mask.unsqueeze(1), mask)
            attns.append(attn)
        
        return enc_output, pad_mask, {"attns":attns, "pos":pos}
    
    def recognize_forward(self, input, pad_mask=None):
        x = self.embedding(input)
        enc_output, pos = self.pos_emb(x)

        # mask = get_tril_mask(input)

        attns = []
        for layer in self.layers:
            enc_output, attn = layer(enc_output)
            attns.append(attn)
        
        return enc_output, pad_mask, {"attns":attns, "pos":pos}