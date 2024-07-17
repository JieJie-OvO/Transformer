import torch
import torch.nn as nn
import torch.nn.functional as F

class JointNet(nn.Module):
    def __init__(self, input_size=640, inner_dim=512, vocab_size=4232):
        super(JointNet, self).__init__()

        self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)

        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size, bias=True)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        
        outputs = self.forward_layer(concat_state)
        outputs = self.tanh(outputs)

        outputs = self.project_layer(outputs)
        return outputs
    
    def recognize_forward(self, enc_state, dec_state):
        assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        
        outputs = self.forward_layer(concat_state)
        outputs = self.tanh(outputs)

        outputs = self.project_layer(outputs)

        return outputs