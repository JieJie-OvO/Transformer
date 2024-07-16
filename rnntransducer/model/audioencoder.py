import torch
import torch.nn as nn

class AudioLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=512, n_layers=6, dropout=0.2):
        super(AudioLSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.output_proj = nn.Linear(2 * hidden_size, output_size, bias=True)

    def forward(self, x):
        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(x)

        logits = self.output_proj(outputs)
        return logits, hidden