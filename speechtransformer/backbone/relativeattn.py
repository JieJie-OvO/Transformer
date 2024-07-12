import math
import logging
import torch
import torch.nn as nn


class RelMultiHeadedSelfAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout_rate=0.0):
        super(RelMultiHeadedSelfAttention, self).__init__()

        self.d_model = d_model
        self.nheads = n_heads
        self.d_k = d_model // n_heads

        self.qvk_proj = nn.Linear(d_model, d_model * 3)

        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.posu = nn.Parameter(torch.Tensor(1, 1, n_heads, self.d_k))
        self.posv = nn.Parameter(torch.Tensor(1, 1, n_heads, self.d_k))

        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def relpos_bias(self, q_v, pos):
        b, _, n, _ = q_v.size()
        S = pos.size(2)
        T = (S+1)//2

        matrix_bd = torch.matmul(q_v.transpose(1, 2), pos.transpose(-2, -1))
        rel_pos = torch.arange(0, T, dtype=torch.long, device=matrix_bd.device)
        rel_pos = (rel_pos[None] - rel_pos[:, None]).reshape(1, 1, T, T) + (T - 1)
        return torch.gather(matrix_bd, dim=3, index=rel_pos.repeat(b, n, 1, 1))

    def forward(self, x, mask, pos):
        x = self.qvk_proj(x)

        query, key, value = torch.split(x, self.d_model, dim=-1)

        batch_size = x.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        
        batch_size = pos.size(0)
        pos = self.pos_proj(pos)
        pos = pos.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)

        q_u = query + self.posu
        q_u = q_u.transpose(1,2)
        q_v = query + self.posv

        AC = torch.matmul(q_u, key.transpose(-2, -1))
        BD = self.relpos_bias(q_v, pos)

        scores = (AC + BD) / math.sqrt(self.d_k)

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
