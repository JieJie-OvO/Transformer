import torch
import torch.nn as nn
import torch.nn.functional as F
from .windowconencoder import WinConformerEncoder
from speechtransformer.frontend import ConvFrontEnd
from .vab import BLK
import math

class LSTMDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, num_layers):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(d_model, d_model*2, num_layers)
        self.fc = nn.Linear(d_model*2, vocab_size)
        
        self.ctc_crit = nn.CTCLoss(blank=BLK, zero_infinity=True)

    def forward(self, x, x_len, tg, tg_len):
        x = x.transpose(0,1)
        x, _ = self.lstm(x)
        x = x.transpose(0,1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)

        loss = self.ctc_crit(x.transpose(0, 1), tg, x_len, tg_len)
        return loss

    def inference(self, x, mask):
        x = x.transpose(0,1)
        x, _ = self.lstm(x)
        x = x.transpose(0,1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)
        length = torch.sum(mask.squeeze(1), dim=-1)

        return x, length


class StreamingModel(nn.Module):
    def __init__(self, fbank=40, channel=[1,64,128], kernel_size=[3,3], 
                 stride=[2,2], vocab_size=4233, d_model=256, n_heads=4, 
                 d_ff=2048, enclayers=12, declayers=6, pre_norm=False, relative = True):
        super(StreamingModel,self).__init__()
        self.frontend = ConvFrontEnd(fbank,d_model, channel, kernel_size, stride)
        self.encoder = WinConformerEncoder(d_model, n_heads, d_ff, enclayers,pre_norm=pre_norm, relative=relative)
        self.decoder = LSTMDecoder(d_model, vocab_size, declayers)

    def forward(self, inputs, targets):
        enc_inputs = inputs['inputs']
        enc_mask = inputs['mask']

        truth = targets['targets']
        truth_length = targets['targets_length']

        enc_inputs, enc_mask = self.frontend(enc_inputs, enc_mask)

        enc_out, enc_mask, _ = self.encoder(enc_inputs, enc_mask)

        enc_length = torch.sum(enc_mask, dim=-1)

        loss = self.decoder(enc_out, enc_length, truth[:, 1:-1], truth_length.add(-1))
        return loss

    def inference(self, enc_inputs, enc_mask):
        enc_inputs, enc_mask = self.frontend(enc_inputs, enc_mask)
        enc_out, enc_mask, _ = self.encoder(enc_inputs, enc_mask)

        return self.decoder.inference(enc_out, enc_mask)

    def save_model(self, path):
        checkpoint = {
            'frontend': self.frontend.state_dict(),
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict()
            }

        torch.save(checkpoint, path)
    
    def load_model(self, chkpt):
        self.frontend.load_state_dict(chkpt['frontend'])
        self.encoder.load_state_dict(chkpt['encoder'])
        self.decoder.load_state_dict(chkpt['decoder'])
    