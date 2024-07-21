import torch
import torch.nn as nn
import torch.nn.functional as F
from .labelencoder import LabelLSTMencoder
from .audioencoder import AudioLSTMEncoder
from .joint import JointNet
from .vab import BLK
from torchaudio.functional import rnnt_loss
from .subsampling import *
import time

class Transducer(nn.Module):
    def __init__(self, fbank=80, input_size=160, enc_hidden=512, enc_out=320, enc_layers=6, 
                 dec_hidden=512, vocab_size=4232, dec_out=320, dec_layers=1, 
                 joint_dim=512,
                 dropout = 0.2, predict_strategy="RNA", lmweight=0.1):
        
        super(Transducer, self).__init__()
        self.subsample = Conv2dSubsampling4(fbank, input_size, 0)
        self.audioencoder = AudioLSTMEncoder(input_size, enc_hidden, enc_out, enc_layers, dropout)
        self.labelencoder = LabelLSTMencoder(dec_hidden, vocab_size, dec_out, n_layers=dec_layers, dropout=dropout)
        self.joint = JointNet(dec_out+enc_out, joint_dim, vocab_size)

        self.joint.project_layer.weight = self.labelencoder.embedding.weight

        self.predict_strategy = predict_strategy
        self.lmweight = lmweight

    def forward(self, inputs_dict, targets_dict):
        inputs = inputs_dict['inputs']
        mask = inputs_dict['mask']
        inputs_length = inputs_dict['sub_inputs_length']
        targets = targets_dict['targets']
        targets_length = targets_dict['targets_length']

        inputs, mask = self.subsample(inputs, mask)

        inputs_length = inputs_length.to('cpu')
        enc_state, _ = self.audioencoder(inputs, inputs_length.int())
        inputs_length = inputs_length.to('cuda')

        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=0)

        targets_length = targets_length.to('cpu')
        dec_state, _ = self.labelencoder(concat_targets, targets_length.add(1).int())
        targets_length = targets_length.to('cuda')

        logits = self.joint(enc_state, dec_state)

        # bs = logits.size(0)
        # steps = bs//4 if bs%4==0 else bs//4 + 1
        # loss_l = []
        # for i in range(steps):
        #     if i == steps-1:
        #         f_len = torch.max(inputs_length[i*4:]).item()
        #         t_len = torch.max(targets_length[i*4:]).item()
        #         loss1 = rnnt_loss(logits[i*4:, 0:f_len, 0:t_len+1, :], targets[i*4:,0:t_len].int(), inputs_length[i*4:].int(), targets_length[i*4:].int(), blank=0)
        #         loss_l.append(loss1)
        #     else:
        #         f_len = torch.max(inputs_length[i*4:i*4+4]).item()
        #         t_len = torch.max(targets_length[i*4:i*4+4]).item()
        #         loss1 = rnnt_loss(logits[i*4:i*4+4, 0:f_len, 0:t_len+1, :], targets[i*4:i*4+4,0:t_len].int(), inputs_length[i*4:i*4+4].int(), targets_length[i*4:i*4+4].int(), blank=0)
        #         loss_l.append(loss1)
        # loss = loss_l[0]
        # for i in range(steps):
        #     if i == 0:
        #         continue
        #     else:
        #         loss += loss_l[i]
        # return loss/steps

        loss = rnnt_loss(logits, targets.int(), inputs_length.int(), targets_length.int(), blank=0)
        return loss


    def recognize(self, inputs_dict, LM=None):
        inputs = inputs_dict['inputs']
        mask = inputs_dict['mask']
        inputs_length = inputs_dict['sub_inputs_length']
        
        batch_size = inputs.size(0)

        inputs, mask = self.subsample(inputs, mask)
        enc_states, _ = self.audioencoder(inputs)

        results = []

        if self.predict_strategy == "RNA":
            for i in range(batch_size):
                decoded_seq = self.rna_decode(enc_states[i], inputs_length[i], LM)
                results.append(decoded_seq)
        else:
            for i in range(batch_size):
                decoded_seq = self.rnnt_decode(enc_states[i], inputs_length[i], LM)
                results.append(decoded_seq)
        
        return results


    def rna_decode(self, enc_state, lengths, LM=None):

        zero_token = torch.LongTensor([[0]])
        if enc_state.is_cuda:
            zero_token = zero_token.cuda()

        token_list = []
        token_list.append(0)

        dec_state, hidden = self.labelencoder(zero_token)

        if LM is not None:
            lm_out = LM(token_list).cuda()

        for t in range(lengths):
            logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
            out = F.softmax(logits, dim=0).detach()
            if LM is not None:
                out = out*(1-self.lmweight) + lm_out*self.lmweight
            pred = torch.argmax(out, dim=0)
            pred = int(pred.item())

            if pred != 0:
                token_list.append(pred)
                token = torch.LongTensor([[pred]])

                if enc_state.is_cuda:
                    token = token.cuda()

                dec_state, hidden = self.labelencoder(token, hidden=hidden)
                if LM is not None:
                    lm_out = LM(token_list).cuda()
        
        res = []
        for i in token_list:
            if i != 0:
                res.append(i)

        return res
    
    def rnnt_decode(self, enc_state, lengths, LM=None):

        zero_token = torch.LongTensor([[0]])
        if enc_state.is_cuda:
            zero_token = zero_token.cuda()

        token_list = []
        token_list.append(0)

        dec_state, hidden = self.labelencoder(zero_token)

        if LM is not None:
            lm_out = LM(token_list).cuda()

        for t in range(lengths):
            logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
            out = F.softmax(logits, dim=0).detach()
            if LM is not None:
                out = out*(1-self.lmweight) + lm_out*self.lmweight
            pred = torch.argmax(out, dim=0)
            pred = int(pred.item())

            if pred != 0:
                token_list.append(pred)
                token = torch.LongTensor([[pred]])

                if enc_state.is_cuda:
                    token = token.cuda()

                t = t-1
                dec_state, hidden = self.labelencoder(token, hidden=hidden)
                if LM is not None:
                    lm_out = LM(token_list).cuda()
                

        res = []
        for i in token_list:
            if i != 0:
                res.append(i)

        return res

    def save_model(self, path):
        checkpoint = {
            "subsample":self.subsample.state_dict(),
            'audioencoder': self.audioencoder.state_dict(),
            'labelencoder': self.labelencoder.state_dict(),
            'joint': self.joint.state_dict()
            }

        torch.save(checkpoint, path)
    
    def load_model(self, chkpt):
        self.subsample.load_state_dict(chkpt['subsample'])
        self.audioencoder.load_state_dict(chkpt['audioencoder'])
        self.labelencoder.load_state_dict(chkpt['labelencoder'])
        self.joint.load_state_dict(chkpt['joint'])
    