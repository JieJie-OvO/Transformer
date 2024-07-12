import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_ACTIVATION = {
    'relu': F.relu,
    'gelu': F.gelu,
    'glu': F.glu,
    'tanh': lambda x: torch.tanh(x),
    'swish': lambda x: x * torch.sigmoid(x)
}


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation='glu'):
        super(PositionwiseFeedForward, self).__init__()
        self.activation = activation

        self.w_1 = nn.Linear(d_model, d_ff * 2 if activation == 'glu' else d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = _ACTIVATION[self.activation](x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x

class Swish(nn.Module):
    def __init__(self, factor=1.0):
        super(Swish, self).__init__()
        self.factor = factor

    def forward(self, x):
        return x * torch.sigmoid(x * self.factor)

class FeedForwardModule(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardModule, self).__init__()
        
        self.sequence = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.sequence(x)