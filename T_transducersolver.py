from Ttransducer.model.t_transducer import T_Transducer
from train.scheduler import TransformerScheduler
from train.trainer import Trainer
from Ttransducer.data.dataloader import FeatureLoader
from Ttransducer.train.utils import map_to_cuda
import editdistance
import torch
import time

class Solver():
    def __init__(self, model, train_wav_path, train_text_path, test_wav_path, 
                 test_text_path, vab_path, fbank = 40, batch_size=16, ngpu=1, 
                 train_epochs = 60, accum_steps=4,
                 lm=None, lm_weight=0.0):
        
        self.model = model
        self.ngpu = ngpu
        self.train_epochs = train_epochs
        self.accum_steps = accum_steps

        if ngpu >= 1:
            model.cuda()
        
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr = 0.001, betas=[0.9,0.98], eps= 1.0e-9, 
                                     weight_decay=1.0e-6, amsgrad= False )

        self.scheduler = TransformerScheduler(self.optimizer, 256, 12000, 1.0)

        self.train_loader = FeatureLoader(train_wav_path, train_text_path, vab_path, fbank, spec_augment=True, ngpu=1, batch_size=batch_size)
        self.test_loader = FeatureLoader(test_wav_path, test_text_path, vab_path, fbank, spec_augment=False, ngpu=1, batch_size=1)

        self.lm = lm
        self.lm_weight = lm_weight

    def train(self):
        self.trainer = Trainer(self.model, self.optimizer, self.scheduler, epochs=self.train_epochs,accum_steps=self.accum_steps)
        self.trainer.train(self.train_loader)

    def load_model(self, path):
        chkpt = torch.load(path)
        self.model.load_model(chkpt)

    def recognize(self):
        idx2unit = self.train_loader.dataset.idx2unit()
        self.model.eval()
        top_n_false_tokens = 0
        false_tokens = 0
        total_tokens = 0
        accu_time = 0

        writer = open("./log/predict.txt", 'w', encoding='utf-8')
        detail_writer = open("./log/predict.log", 'w', encoding='utf-8')

        for step, (utt_id, inputs, targets) in enumerate(self.test_loader.loader):
            if self.ngpu>0:
                inputs = map_to_cuda(inputs)

            st = time.time()
            preds = self.model.recognize(inputs)
            et = time.time()
            span = et - st
            accu_time += span

            totals = len(self.test_loader.loader)

            truths = targets['targets']
            truths_length = targets['targets_length']

            for b in range(len(preds)):
                n = step + b

                truth = [idx2unit[i.item()] for i in truths[b][:truths_length[b]]]
                truth = ' '.join(truth)

                print_info = '[%d / %d ] %s - truth: %s' % (n, totals, utt_id[b], truth)
                detail_writer.write(print_info+'\n')
                total_tokens += len(truth.split())  

                nbest_min_false_tokens = 1e10
                pred = preds[b]

                pred_str = [idx2unit[i] for i in pred]
                pred = ' '.join(pred_str)

                _truth = truth.replace("<PESN> ", "").replace("<VIET> ", "").replace("<SWAH> ", "")
                _pred = pred.replace("<PESN> ", "").replace("<VIET> ", "").replace("<SWAH> ", "")
                n_diff = editdistance.eval(_truth.split(), _pred.split())

                false_tokens += n_diff
                nbest_min_false_tokens = min(nbest_min_false_tokens, n_diff)

                print_info = '[%d / %d ] %s - pred : %s' % (n, totals, utt_id[b], pred)
                detail_writer.write(print_info+'\n')
                
                writer.write(utt_id[b] + ' ' + pred + '\n')
                top_n_false_tokens += nbest_min_false_tokens

                detail_writer.write('\n')

        writer.close()
        detail_writer.close()

        with open("./log/result.txt", 'w', encoding='utf-8') as w:
            cer = false_tokens / total_tokens * 100
            w.write('The CER is %.3f. \n' % cer)

fbank=80
d_model=256
n_heads=4
d_ff=2048
audio_layers=6
vocab_size=4232
label_layers=3
inner_dim=2048
dropout=0.1
pre_norm=False
chunk_size = 15
predict_strategy="RNA"


train_wav_path = "egs/aishell/data/train/wav.scp"
train_text_path = "egs/aishell/data/train/text"
test_wav_path = "egs/aishell/data/test/wav.scp"
test_text_path = "egs/aishell/data/test/text"
vab_path = "egs/aishell/data/transducer_vab"
batch_size = 16
train_epochs = 80
accum_steps = 4

ngpu = 1 if torch.cuda.is_available() else 0
print("ngpu: ", ngpu)

model = T_Transducer(fbank, d_model, n_heads, d_ff, audio_layers, 
                     vocab_size, label_layers, 
                     inner_dim,
                     dropout, pre_norm, chunk_size,
                     predict_strategy=predict_strategy)

solver = Solver(model, train_wav_path,train_text_path, test_wav_path, test_text_path,
                vab_path, fbank, batch_size, ngpu, train_epochs = train_epochs, accum_steps=accum_steps)

# solver.train()
solver.load_model("./result/Ttransducer/model.epoch.69.pth")
solver.recognize()
# 133250