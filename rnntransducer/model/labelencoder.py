import torch
import torch.nn as nn


class LabelLSTMencoder(nn.Module):
    def __init__(self, hidden_size=512, vocab_size=4232, output_size=320, n_layers=1, dropout=0.2, share_weight=False):
        super(LabelLSTMencoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.output_proj = nn.Linear(hidden_size, output_size)

        if share_weight:
            self.embedding.weight = self.output_proj.weight

    def forward(self, inputs, hidden=None):

        embed_inputs = self.embedding(inputs)

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(embed_inputs, hidden)

        outputs = self.output_proj(outputs)

        return outputs, hidden
