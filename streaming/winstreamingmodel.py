import torch
import torch.nn as nn
import torch.nn.functional as F
from .streamingdecoder import StreamingDecoder
from .streamingencoder import StreamingEncoder
from speechtransformer.frontend import ConvFrontEnd
from .vab import BLK
import math

class StreamingModel(nn.Module):
    def __init__(self, fbank=40, channel=[1,64,128], kernel_size=[3,3], 
                 stride=[2,2], vocab_size=4233, d_model=256, n_heads=4, 
                 d_ff=2048, enclayers=12, declayers=6, pre_norm=False, relative = True,
                 windows = 100):
        super(StreamingModel,self).__init__()
        self.frontend = ConvFrontEnd(fbank,d_model, channel, kernel_size, stride)
        self.encoder = StreamingEncoder(d_model, n_heads, d_ff, enclayers,pre_norm=pre_norm, relative=relative)
        self.decoder = StreamingDecoder()
        self.windows = windows

        self.ctc_crit = nn.CTCLoss(blank=BLK, zero_infinity=True)

        self.d_model = d_model
        self.declayers = declayers

    def forward(self, inputs, targets):
        enc_inputs = inputs['inputs']
        enc_mask = inputs['mask']

        truth = targets['targets']
        truth_length = targets['targets_length']

        enc_inputs, enc_mask = self.frontend(enc_inputs, enc_mask)

        caches1 = []
        caches2 = []

        x = None
        last_mask = None
        _,len,_ = enc_inputs.size()
        i = 0
        while i < len:
            if i + self.windows < len:
                sub_enc_inputs = enc_inputs[:,i:i+self.windows,:]
                sub_enc_mask = enc_mask[:, i:i+self.windows]
            else:
                sub_enc_inputs = enc_inputs[:,i:,:]
                sub_enc_mask = enc_mask[:, i:]
            
            
            enc_out, sub_enc_mask, caches1 = self.encoder(sub_enc_inputs, caches1, sub_enc_mask, last_mask)
            new_x,sub_enc_mask,caches2 = self.decoder(enc_out, caches2, sub_enc_mask, last_mask)

            if x is None:
                x = new_x
            else:
                x = torch.cat((x, new_x),dim=1)

            i = i + self.windows
            last_mask = sub_enc_mask

        enc_length = torch.sum(enc_mask, dim=-1)

        loss = self.ctc_crit(x.transpose(0, 1), truth[:, 1:-1], enc_length, truth_length.add(-1))
        return loss

    def inference(self, enc_inputs, enc_mask):
        enc_inputs, enc_mask = self.frontend(enc_inputs, enc_mask)
        length = torch.sum(enc_mask.squeeze(1), dim=-1)
        caches1 = []
        caches2 = []

        x = None
        last_mask = None
        _,len,_ = enc_inputs.size()
        i = 0
        while i < len:
            # print(i,self.windows)
            if i + self.windows < len:
                sub_enc_inputs = enc_inputs[:,i:i+self.windows,:]
                sub_enc_mask = enc_mask[:, i:i+self.windows]
            else:
                sub_enc_inputs = enc_inputs[:,i:,:]
                sub_enc_mask = enc_mask[:, i:]
            
            
            enc_out, sub_enc_mask, caches1 = self.encoder(sub_enc_inputs, caches1, sub_enc_mask, last_mask)
            new_x,sub_enc_mask,caches2 = self.decoder(enc_out,caches2,sub_enc_mask, last_mask)

            if x is None:
                x = new_x
            else:
                x = torch.cat((x, new_x),dim=1)

            i = i + self.windows
            last_mask = sub_enc_mask

        return x, length

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
    