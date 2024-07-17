import torch
import torch.nn as nn
import torch.nn.functional as F
from .labelencoder import LabelLSTMencoder
from .audioencoder import AudioLSTMEncoder
from .joint import JointNet
from .vab import BLK
from torchaudio.functional import rnnt_loss

class Transducer(nn.Module):
    def __init__(self, input_size=80, enc_hidden=512, enc_out=340, enc_layers=6, 
                 dec_hidden=512, vocab_size=4232, dec_out=320, dec_layers=1, 
                 joint_dim=512,
                 dropout = 0.2, predict_strategy="RNA"):
        
        super(Transducer, self).__init__()
        self.audioencoder = AudioLSTMEncoder(input_size, enc_hidden, enc_out, enc_layers, dropout)
        self.labelencoder = LabelLSTMencoder(dec_hidden, vocab_size, dec_out, n_layers=dec_layers, dropout=dropout)
        self.joint = JointNet(dec_out+enc_out, joint_dim, vocab_size)

        self.joint.project_layer.weight = self.labelencoder.embedding.weight

        self.predict_strategy = predict_strategy


    def forward(self, inputs_dict, targets_dict):
        inputs = inputs_dict['inputs']
        inputs_length = inputs_dict['inputs_length']

        targets = targets_dict['targets']
        targets_length = targets_dict['targets_length']

        enc_state, _ = self.audioencoder(inputs)
        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=0)

        dec_state, _ = self.labelencoder(concat_targets)

        logits = self.joint(enc_state, dec_state)

        # logits = logits.to('cpu')
        # targets = targets.to('cpu')
        # inputs_length = inputs_length.to('cpu')
        # targets_length = targets_length.to('cpu')
        loss = rnnt_loss(logits, targets.int(), inputs_length.int(), targets_length.int(), blank=0)
        return loss
    
    def recognize(self, inputs_dict):
        inputs = inputs_dict['inputs']
        inputs_length = inputs_dict['inputs_length']
        
        batch_size = inputs.size(0)

        enc_states, _ = self.audioencoder(inputs)

        results = []

        if self.predict_strategy == "RNA":
            for i in range(batch_size):
                decoded_seq = self.rna_decode(enc_states[i], inputs_length[i])
                results.append(decoded_seq)
        else:
            for i in range(batch_size):
                decoded_seq = self.rnnt_decode(enc_states[i], inputs_length[i])
                results.append(decoded_seq)
        
        return results


    def rna_decode(self, enc_state, lengths):

        zero_token = torch.LongTensor([[0]])
        if enc_state.is_cuda:
            zero_token = zero_token.cuda()

        token_list = []

        dec_state, hidden = self.labelencoder(zero_token)

        for t in range(lengths):
            logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
            out = F.softmax(logits, dim=0).detach()
            pred = torch.argmax(out, dim=0)
            pred = int(pred.item())

            if pred != 0:
                token_list.append(pred)
                token = torch.LongTensor([[pred]])

                if enc_state.is_cuda:
                    token = token.cuda()

                dec_state, hidden = self.labelencoder(token, hidden=hidden)

        return token_list
    
    def rnnt_decode(self, enc_state, lengths):

        zero_token = torch.LongTensor([[0]])
        if enc_state.is_cuda:
            zero_token = zero_token.cuda()

        token_list = []

        dec_state, hidden = self.labelencoder(zero_token)

        for t in range(lengths):
            logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
            out = F.softmax(logits, dim=0).detach()
            pred = torch.argmax(out, dim=0)
            pred = int(pred.item())

            if pred != 0:
                token_list.append(pred)
                token = torch.LongTensor([[pred]])

                if enc_state.is_cuda:
                    token = token.cuda()

                t = t-1
                dec_state, hidden = self.labelencoder(token, hidden=hidden)

        return token_list

    def save_model(self, path):
        checkpoint = {
            'audioencoder': self.audioencoder.state_dict(),
            'labelencoder': self.labelencoder.state_dict(),
            'joint': self.joint.state_dict()
            }

        torch.save(checkpoint, path)
    
    def load_model(self, chkpt):
        self.audioencoder.load_state_dict(chkpt['audioencoder'])
        self.labelencoder.load_state_dict(chkpt['labelencoder'])
        self.joint.load_state_dict(chkpt['joint'])
    