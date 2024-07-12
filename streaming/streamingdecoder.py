import torch
import torch.nn as nn
import torch.nn.functional as F
from speechtransformer.backbone.pos import PositionalEncoding
from speechtransformer.backbone.feedforward import PositionwiseFeedForward
from .winattention import XLMultiHeadedSelfAttention
from speechtransformer.backbone.pos import PositionalEncoding

class StreamingEncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, slf_attn_drop=0.0,
                 ffn_drop=0.0, residual_dropout=0.1, pre_norm=False, relative = False):
        
        super(StreamingEncoderLayer,self).__init__()
        self.attn = XLMultiHeadedSelfAttention(n_heads, d_model, slf_attn_drop)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, ffn_drop)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(residual_dropout)
        self.dropout2 = nn.Dropout(residual_dropout)

        self.pre_norm = pre_norm
        self.relative = relative

    def forward(self, x1, x2=None, mask=None, mask2=None):
        if x2 is None:
            x2 = x1
        else:
            x2 = torch.cat([x2,x1],dim=1)

        residual = x1
        if self.pre_norm:
            x = self.norm1(x1)
        
        x,_ = self.attn(x1, x2, mask.unsqueeze(1), mask2)
        x = self.dropout1(x) + residual

        if not self.pre_norm:
            x = self.norm1(x)

        residual = x
        if self.pre_norm:
            x = self.norm2(x)

        x = self.feed_forward(x)
        x = residual + self.dropout2(x)

        if not self.pre_norm:
            x = self.norm2(x)
        
        return x, None
    

class StreamingDecoder(nn.Module):
    def __init__(self, vocab_size=4233, d_model=256, n_heads=4, d_ff=2048, enc_dim=256, n_layers=6, attn_drop=0.0,
                  ffn_drop=0.0, residual_drop=0.1, pre_norm=False, relative = False):
        super(StreamingDecoder,self).__init__()

        self.relative = relative
        self.pre_norm = pre_norm

        self.pos_emb = PositionalEncoding(d_model)

        self.layers = nn.ModuleList()

        for i in range(n_layers):
            self.layers.append(StreamingEncoderLayer(n_heads, d_model, d_ff, attn_drop, ffn_drop, residual_drop, pre_norm, relative=relative))

        if pre_norm:
            self.norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, caches, mask, lastmask=None):

        if self.relative == False:
            enc_output, pos = self.pos_emb(inputs)
        else:
            enc_output = inputs
            if len(caches)!=0:
                position = torch.arange(-(inputs.size(1)*2-1), inputs.size(1)*2, device=inputs.device).reshape(1, -1)
            else:
                position = torch.arange(-(inputs.size(1)-1), inputs.size(1), device=inputs.device).reshape(1, -1)
            pos = self.pos_emb._embedding_from_positions(position)

        new_caches = []
        for (i,layer) in enumerate(self.layers):
            if len(caches)==0:
                enc_output, attn = layer(enc_output,None, mask, lastmask)
            else:
                enc_output, attn = layer(enc_output,caches[i], mask, lastmask)
            new_caches.append(enc_output)

        if self.pre_norm:
            enc_output = self.norm(enc_output)

        enc_output = self.fc(enc_output)

        return F.log_softmax(enc_output, dim=-1), mask, new_caches
