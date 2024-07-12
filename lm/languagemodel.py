import torch
import torch.nn as nn
from speechtransformer.encoder import EncoderLayer
from speechtransformer.backbone.pos import PositionalEncoding
from speechtransformer.backbone.loss import LabelSmoothingLoss
import torch.nn.functional as F

def get_seq_mask(targets):
    batch_size, steps = targets.size()
    seq_mask = torch.ones([batch_size, steps, steps], device=targets.device)
    seq_mask = torch.tril(seq_mask).bool()
    return seq_mask

class LM(nn.Module):
    def __init__(self, vocab_size=4233, nheads=4, d_model=256, d_ff=2048, numlayers=6):
        super(LM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_embedding = PositionalEncoding(d_model)

        self.lm_blocks = nn.ModuleList()

        for i in range(numlayers):
            self.lm_blocks.append(EncoderLayer(nheads,d_model,d_ff))

        self.out_linear = nn.Linear(d_model, vocab_size)

        self.out_linear.weight = self.embedding.weight

        self.cirt = LabelSmoothingLoss(vocab_size, 0.1)
        
    def forward(self, inputs, targets):
        x = inputs['inputs']
        x = self.embedding(x)
        mask = get_seq_mask(inputs['inputs'])
        x,_ = self.pos_embedding(x)

        for block in self.lm_blocks:
            x,_ = block(x,mask)
        
        x = self.out_linear(x)

        gt = targets['targets']
        loss = self.cirt(x,gt)

        return loss
    
    def predict(self, targets, last_frame=True):
        dec_output = self.embedding(targets)
        dec_output, _ = self.pos_embedding(dec_output)

        dec_mask = get_seq_mask(targets)

        for _, block in enumerate(self.lm_blocks):
            dec_output, _ = block(dec_output, dec_mask)

        logits = self.out_linear(dec_output)

        if last_frame:
            log_probs = F.log_softmax(logits[:, -1, :].unsqueeze(1), dim=-1)
        else:
            log_probs = F.log_softmax(logits, dim=-1)

        return log_probs
    
    def save_model(self, path):
        checkpoint = {
            'model': self.state_dict()
            }
        torch.save(checkpoint, path)
    
    def load_model(self, chkpt):
        self.load_state_dict(chkpt['model'])