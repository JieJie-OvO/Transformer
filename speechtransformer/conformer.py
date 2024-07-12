import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import TransformerDecoder
from .conformerencoder import ConformerEncoder
from .frontend import ConvFrontEnd
from .backbone.loss import LabelSmoothingLoss


class Conformer(nn.Module):
    def __init__(self, fbank=40, channel=[1,64,128], kernel_size=[3,3], 
                 stride=[2,2], vocab_size=4233, d_model=256, n_heads=4, 
                 d_ff=2048, enclayers=12, declayers=6, pre_norm=False, relative = True):
        super(Conformer,self).__init__()
        self.frontend = ConvFrontEnd(fbank,d_model, channel, kernel_size, stride)
        self.encoder = ConformerEncoder(d_model, n_heads, d_ff, enclayers,pre_norm=pre_norm, relative=relative)
        self.decoder = TransformerDecoder(vocab_size,d_model, n_heads, d_ff, d_model, declayers, pre_norm=pre_norm)

        self.crit = LabelSmoothingLoss(vocab_size)

    def forward(self, inputs, targets):
        enc_inputs = inputs['inputs']
        enc_mask = inputs['mask']

        truth = targets['targets']
        truth_length = targets['targets_length']

        enc_inputs, enc_mask = self.frontend(enc_inputs, enc_mask)

        enc_out, enc_mask, _ = self.encoder(enc_inputs, enc_mask)

        target_in = truth[:, :-1].clone()
        logits, _ = self.decoder(target_in, enc_out, enc_mask)

        target_out = truth[:, 1:].clone()
        loss = self.crit(logits, target_out)

        return loss

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
    