import math
import torch
import torch.nn as nn

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout_rate=0.0):
        super(MultiHeadedSelfAttention, self).__init__()

        self.d_model = d_model
        self.nheads = n_heads
        self.d_k = d_model // n_heads

        self.qvk_proj = nn.Linear(d_model, d_model * 3)

        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        x = self.qvk_proj(x)

        query, key, value = torch.split(x, self.d_model, dim=-1)

        batch_size = x.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        if mask is not None:
            upmask = mask.unsqueeze(1)
            scores.masked_fill_(~upmask, -float('inf'))

        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, value)

        if context.dim() == 4:
            b, n, t, v = context.size()
            context = context.transpose(1, 2).reshape(b, t, n * v)
        
        context = self.output_proj(context)

        context = self.dropout(context)

        return context, weights


class MultiHeadedCrossAttention(nn.Module):
    def __init__(self, n_heads, d_model, enc_dim, dropout_rate=0.0):
        super(MultiHeadedCrossAttention, self).__init__()

        self.d_model = d_model
        self.nheads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.vk_proj = nn.Linear(enc_dim, d_model * 2)

        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, query, enc_out, mask):

        query = self.q_proj(query)
        enc_out = self.vk_proj(enc_out)

        key, value = torch.split(enc_out, self.d_model, dim=-1)

        batch_size = query.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        if mask is not None:
            upmask = mask.unsqueeze(1)
            scores.masked_fill_(~upmask, -float('inf'))

        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, value)

        if context.dim() == 4:
            b, n, t, v = context.size()
            context = context.transpose(1, 2).reshape(b, t, n * v)
        
        context = self.output_proj(context)

        context = self.dropout(context)
        
        return context, weights
    
