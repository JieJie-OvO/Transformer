from train.scheduler import TransformerScheduler
from speechtransformer.speech_transformer import SpeechTransformer
from speechtransformer.conformer import Conformer
import torch
from lm.languagemodel import LM
from data.textloader import TextLoader
from train.lmtrainer import LMTrainer
from recognize.recognizer import Recognizer
from train.trainer import Trainer
from data.dataloader import FeatureLoader
from train.utils import map_to_cuda
import editdistance
import time

class Solver():
    def __init__(self, model, train_wav_path, train_text_path, test_wav_path, 
                 test_text_path, vab_path, fbank = 40, batch_size=16, ngpu=1, 
                 train_epochs = 60, accum_steps=4,
                 lm=None, lm_weight=0.1):
        
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
        self.recognizer = Recognizer(self.model, idx2unit=idx2unit, lm=self.lm, lm_weight=self.lm_weight)
        self.model.eval()
        top_n_false_tokens = 0
        false_tokens = 0
        total_tokens = 0
        accu_time = 0
        total_frames = 0

        writer = open("./log/predict.txt", 'w', encoding='utf-8')
        detail_writer = open("./log/predict.log", 'w', encoding='utf-8')

        for step, (utt_id, inputs, targets) in enumerate(self.test_loader.loader):
            if self.ngpu>0:
                inputs = map_to_cuda(inputs)

            enc_inputs = inputs['inputs']
            enc_mask = inputs['mask']
            total_frames += enc_inputs.size(1)

            st = time.time()
            preds, scores = self.recognizer.recognize(enc_inputs, enc_mask)
            et = time.time()
            span = et - st
            accu_time += span

            totals = len(self.test_loader.loader)

            truths = targets['targets']
            truths_length = targets['targets_length']

            for b in range(len(preds)):
                n = step + b

                truth = [idx2unit[i.item()] for i in truths[b][1:truths_length[b]]]
                truth = ' '.join(truth)

                print_info = '[%d / %d ] %s - truth: %s' % (n, totals, utt_id[b], truth)
                detail_writer.write(print_info+'\n')
                total_tokens += len(truth.split())  

                nbest_min_false_tokens = 1e10
                for i in range(len(preds[b])):
                    pred = preds[b][i]

                    _truth = truth.replace("<PESN> ", "").replace("<VIET> ", "").replace("<SWAH> ", "")
                    _pred = pred.replace("<PESN> ", "").replace("<VIET> ", "").replace("<SWAH> ", "")
                    n_diff = editdistance.eval(_truth.split(), _pred.split())
                    if i == 0:
                        false_tokens += n_diff
                    nbest_min_false_tokens = min(nbest_min_false_tokens, n_diff)

                    print_info = '[%d / %d ] %s - pred : %s' % (n, totals, utt_id[b], pred)
                    detail_writer.write(print_info+'\n')
                
                writer.write(utt_id[b] + ' ' + preds[b][0] + '\n')
                top_n_false_tokens += nbest_min_false_tokens

                detail_writer.write('\n')

        writer.close()
        detail_writer.close()

        with open("./log/result.txt", 'w', encoding='utf-8') as w:
            cer = false_tokens / total_tokens * 100
            w.write('The CER is %.3f. \n' % cer)

fbank=40
channel=[1,64,128]
kernel_size=[3,3]
stride=[2,2]
vocab_size=4233
d_model=256
n_heads=4
d_ff=2048
enclayers=12
declayers=6
pre_norm = False
relative = True # just for conformer

train_wav_path = "egs/aishell/data/train/wav.scp"
train_text_path = "egs/aishell/data/train/text"
test_wav_path = "egs/aishell/data/test/wav.scp"
test_text_path = "egs/aishell/data/test/text"
vab_path = "egs/aishell/data/vocab"
batch_size = 64
train_epochs = 60
accum_steps = 1
lm_weight = 0.1

ngpu = 1 if torch.cuda.is_available() else 0
print("ngpu: ", ngpu)

# 关于languemodel
lm = LM(vocab_size,n_heads,d_model,d_ff,6)
lm_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, lm.parameters()),
                                     lr = 0.0001, betas=[0.9,0.98], eps= 1.0e-9, 
                                     weight_decay=1.0e-6, amsgrad= False )
lm_scheduler = TransformerScheduler(lm_optimizer, 256, 12000, 1.0)
chkpt = torch.load("./model.epoch.59.pth")
lm.load_model(chkpt)
textloader = TextLoader(train_text_path, vab_path, ngpu, batch_size)
lmtrainer = LMTrainer(lm, lm_optimizer, lm_scheduler, train_epochs, accum_steps, keep_last_n_chkpt=3, ngpu=ngpu)
lmtrainer.train(textloader)

model_name = 1

if model_name == 0:
    model = SpeechTransformer(fbank, channel, kernel_size, stride, vocab_size, d_model,                           
                            n_heads, d_ff, enclayers, declayers, pre_norm)
else:
    model = Conformer(fbank, channel, kernel_size, stride, vocab_size, d_model,
                    n_heads, d_ff, enclayers, declayers, pre_norm, relative=relative)


solver = Solver(model, train_wav_path,train_text_path, test_wav_path, test_text_path,
                vab_path, fbank, batch_size, ngpu, train_epochs = train_epochs, accum_steps=accum_steps, lm=lm, lm_weight= lm_weight)

solver.load_model("./result/relconformer/relconformer.pth")
solver.recognize()
