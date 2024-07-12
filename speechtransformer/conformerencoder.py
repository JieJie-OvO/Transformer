import torch
import torch.nn as nn
from .backbone.pos import PositionalEncoding
from .backbone.feedforward import PositionwiseFeedForward, FeedForwardModule
from .backbone.attention import MultiHeadedSelfAttention
from .backbone.convblock import ConformerConvModule
from .backbone.relativeattn import RelMultiHeadedSelfAttention

class ConformerEncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, slf_attn_drop=0.0, conv_drop=0.0, 
                 ffn_drop=0.0, residual_dropout=0.1, pre_norm=False, relative = False):
        super(ConformerEncoderLayer, self).__init__()

        self.feed_forward1 = FeedForwardModule(d_model, d_ff, ffn_drop)
        self.conv = ConformerConvModule(d_model, dropout_p=conv_drop)
        if relative:
            self.attn = RelMultiHeadedSelfAttention(n_heads, d_model, slf_attn_drop)
        else:
            self.attn = MultiHeadedSelfAttention(n_heads, d_model, slf_attn_drop)
        self.feed_forward2 = FeedForwardModule(d_model, d_ff, ffn_drop)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(residual_dropout)
        self.dropout2 = nn.Dropout(residual_dropout)
        self.dropout3 = nn.Dropout(residual_dropout)
        self.dropout4 = nn.Dropout(residual_dropout)

        self.pre_norm = pre_norm
        self.relative = relative


    def forward(self, x, mask = None, pos =None):

        residual = x
        if self.pre_norm:
            x = self.norm1(x)
        
        x = self.feed_forward1(x)
        x = residual + 0.5*self.dropout1(x)

        if not self.pre_norm:
            x = self.norm1(x)

        residual = x
        if self.pre_norm:
            x = self.norm2(x)

        if self.relative:
            x, attn = self.attn(x, mask.unsqueeze(1), pos)
        else:
            x, attn = self.attn(x, mask.unsqueeze(1))

        x = residual + self.dropout1(x)

        if not self.pre_norm:
            x = self.norm2(x)

        residual = x
        if self.pre_norm:
            x = self.norm3(x)
        
        x = self.conv(x, mask)
        x = residual + self.dropout3(x)

        if not self.pre_norm:
            x = self.norm3(x)

        residual = x
        if self.pre_norm:
            x = self.norm2(x)
        
        x = self.feed_forward2(x)
        x = residual + 0.5*self.dropout4(x)

        if not self.pre_norm:
            x = self.norm2(x)

        return x, attn
    

class ConformerEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=2048, n_layers=12, attn_drop=0.0, conv_drop=0.0,
                  ffn_drop=0.0, residual_drop=0.1, pre_norm=False, relative = False):
        
        super(ConformerEncoder, self).__init__()

        self.relative = relative
        self.pre_norm = pre_norm

        self.pos_emb = PositionalEncoding(d_model)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(ConformerEncoderLayer(n_heads, d_model, d_ff, attn_drop, conv_drop, ffn_drop, residual_drop, pre_norm, relative=relative))

        if pre_norm:
            self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs, mask):
        if self.relative == False:
            enc_output, pos = self.pos_emb(inputs)
        else:
            enc_output = inputs
            position = torch.arange(-(inputs.size(1)-1), inputs.size(1), device=inputs.device).reshape(1, -1)
            pos = self.pos_emb._embedding_from_positions(position)

        attns = []
        for layer in self.layers:
            enc_output, attn = layer(enc_output, mask, pos)
            attns.append(attn)
            
        if self.pre_norm:
            enc_output = self.norm(enc_output)

        return enc_output, mask, {"attns":attns, "pos":pos}