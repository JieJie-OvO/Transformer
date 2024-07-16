import torch
import torch.nn as nn
import torch.nn.functional as F
from .labelencoder import LabelEncoder
from .audioencoder import AudioEncoder
from .jointnet import JointNet
from .vab import BLK
from torchaudio.functional import rnnt_loss

class T_Transducer(nn.Module):
    def __init__(self, fbank=80, d_model=256, n_heads=4, d_ff=2048, audio_layers=6, 
                 vocab_size=4232, label_layers=3, 
                 inner_dim=2048, 
                 dropout=0.1, pre_norm=False, chunk_size = 10):
        super(T_Transducer, self).__init__()

        self.audioencoder = AudioEncoder(fbank, d_model, n_heads, d_ff, audio_layers, 
                                         residual_drop=dropout, pre_norm=pre_norm, 
                                         chunk_size=chunk_size)

        self.labelencoder = LabelEncoder(vocab_size, d_model, n_heads, d_ff, label_layers,
                                         residual_drop=dropout, pre_norm=pre_norm)
        
        self.jointnet = JointNet(d_model*2, inner_dim, vocab_size)

        # self.jointnet.project_layer.weight = self.labelencoder.embedding.weight

    def forward(self, inputs_dict, targets_dict):
        inputs = inputs_dict['inputs']
        inputs_length = inputs_dict['inputs_length']
        input_pad_mask = inputs_dict['mask']

        targets = targets_dict['targets']
        targets_length = targets_dict['targets_length']
        tg_mask = targets_dict['mask']

        enc_state, _, _ = self.audioencoder(inputs, input_pad_mask)

        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=0)

        dec_state, _, _ = self.labelencoder(concat_targets,tg_mask)

        logits = self.jointnet(enc_state, dec_state)

        loss = rnnt_loss(logits, targets.int(), inputs_length.int(), targets_length.int(), blank=0)
        return loss
    
    def recognize(self, inputs_dict):
        inputs = inputs_dict['inputs']
        inputs_length = inputs_dict['inputs_length']
        pad_mask = inputs_dict['mask']

        with torch.no_grad():
            batch_size = inputs.size(0)
            enc_state, _, _ = self.audioencoder(inputs, pad_mask)

            batch_size = inputs.size(0)

            res = []

            for i in range(batch_size):
                res.append(self.decode(enc_state[i], inputs_length[i]))
            
            return res
    
    def decode(self, enc_state, len):
        label = torch.LongTensor([[0]]).to(enc_state.device)
        enc_state = enc_state.unsqueeze(0)

        preds = []

        for i in range(len):
            dec_state, _, _ = self.labelencoder.recognize_forward(label)

            if i == len-1:
                logits = self.jointnet.recognize_forward(dec_state, enc_state[:,:,:])
            else:
                logits = self.jointnet.recognize_forward(dec_state, enc_state[:, :1+i,:])
            
            out = F.softmax(logits[0][-1], dim=-1).detach()
            pred = torch.argmax(out, dim=0)
            pred = int(pred.item())
            preds.append(pred)
            pred = torch.LongTensor([[pred]]).to(enc_state.device)
            label = torch.cat([label, pred], dim = -1)

        results = []
        for i in preds:
            if i != 0:
                results.append(i)
        
        return results

    def save_model(self, path):
        checkpoint = {
            'audioencoder': self.audioencoder.state_dict(),
            'labelencoder': self.labelencoder.state_dict(),
            'joint': self.jointnet.state_dict()
            }

        torch.save(checkpoint, path)
    
    def load_model(self, chkpt):
        self.audioencoder.load_state_dict(chkpt['audioencoder'])
        self.labelencoder.load_state_dict(chkpt['labelencoder'])
        self.jointnet.load_state_dict(chkpt['joint'])