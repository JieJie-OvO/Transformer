from recognize.streamingrecognizer import StreamingRecognizer
from train.scheduler import TransformerScheduler
from train.trainer import Trainer
from streaming.winstreamingmodel import StreamingModel
from data.dataloader import FeatureLoader
from train.utils import map_to_cuda
import editdistance
import torch
import time
from Eval_latency.read_textgrid import textgrid2outlist
from Eval_latency.tools import compute_frames

class Solver():
    def __init__(self, model, train_wav_path, train_text_path, test_wav_path, 
                 test_text_path, vab_path, fbank = 40, batch_size=16, ngpu=1, 
                 train_epochs = 60, accum_steps=4,
                 lm=None, lm_weight=0.0,
                 pinyin_dic = "eval_latency/out", latency_wav_path="Eval_latency/test/wav.scp",
                 latency_text_path="Eval_latency/test/text"):
        
        self.model = model
        self.ngpu = ngpu
        self.train_epochs = train_epochs
        self.accum_steps = accum_steps
        self.pinyin_dic = pinyin_dic

        if ngpu >= 1:
            model.cuda()
        
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr = 0.001, betas=[0.9,0.98], eps= 1.0e-9, 
                                     weight_decay=1.0e-6, amsgrad= False )

        self.scheduler = TransformerScheduler(self.optimizer, 256, 12000, 1.0)

        self.train_loader = FeatureLoader(train_wav_path, train_text_path, vab_path, fbank, spec_augment=True, ngpu=1, batch_size=batch_size)
        self.test_loader = FeatureLoader(test_wav_path, test_text_path, vab_path, fbank, spec_augment=False, ngpu=1, batch_size=1)
        self.latency_loader = FeatureLoader(latency_wav_path, latency_text_path, vab_path, fbank, spec_augment=False, ngpu=1, batch_size=1)

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
        self.recognizer = StreamingRecognizer(self.model, idx2unit=idx2unit, lm=self.lm, lm_weight=self.lm_weight)
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
            preds = self.recognizer.recognize(enc_inputs, enc_mask)
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
                pred = preds[b]

                _truth = truth.replace("<PESN> ", "").replace("<VIET> ", "").replace("<SWAH> ", "")
                _pred = pred.replace("<PESN> ", "").replace("<VIET> ", "").replace("<SWAH> ", "")
                n_diff = editdistance.eval(_truth.split(), _pred.split())

                false_tokens += n_diff
                nbest_min_false_tokens = min(nbest_min_false_tokens, n_diff)

                print_info = '[%d / %d ] %s - pred : %s' % (n, totals, utt_id[b], pred)
                detail_writer.write(print_info+'\n')
                
                writer.write(utt_id[b] + ' ' + preds[b] + '\n')
                top_n_false_tokens += nbest_min_false_tokens

                detail_writer.write('\n')

        writer.close()
        detail_writer.close()

        with open("./log/result.txt", 'w', encoding='utf-8') as w:
            cer = false_tokens / total_tokens * 100
            w.write('The CER is %.3f. \n' % cer)

    def eval_latency(self):
        idx2unit = self.train_loader.dataset.idx2unit()
        reses = textgrid2outlist(self.pinyin_dic)
        self.model.eval()
        accu_time = 0
        total_frames = 0

        latencys = []

        for step, (utt_id, inputs, targets) in enumerate(self.latency_loader.loader):
            latency = []
            if self.ngpu>0:
                inputs = map_to_cuda(inputs)
            total_frames = inputs['inputs'].size(1)

            if utt_id[0] not in reses:
                continue
            res = reses[utt_id[0]]
        
            st = time.time()
            preds = self.model.recognize_with_latency(inputs)

            if step == 0:
                # 最开始跳过，系统还没有稳定
                continue

            pred = preds[0]

            if len(res) != len(pred) + 1:
                continue

            for i in range(len(pred)):
                # [[now, t], pred]
                idx = compute_frames(total_frames, pred[i][0][1])
                et = pred[i][0][0]
                delta = idx * 0.01 + et - st
                min = delta - res[i][1]
                max = delta - res[i][2]
                latency.append([idx2unit[pred[i][1]],min, max])
            
            latency.append(res[-1])
            latencys.append(latency)
        
        FTD = []
        LTD = []
        AvgTD = []
        for latency in latencys:
            sub_list = []
            for i in range(len(latency)-1):
                sub_list.append((latency[i][1]+latency[i][-1])/2 * 1000)
            FTD.append(sub_list[0])
            LTD.append(sub_list[-1])
            avg = 0.
            for i in sub_list:
                avg += i
            AvgTD.append(avg / len(sub_list))
        
        idx50 = int(len(FTD) * 0.5)
        idx90 = int(len(FTD) * 0.9)
        FTD.sort()
        LTD.sort()
        AvgTD.sort()

        with open("./log/latency.txt", 'w', encoding='utf-8') as w:
            w.write('FTD50  :{0}ms        FTD90  :{1}ms\n'.format(FTD[idx50], FTD[idx90]))
            w.write('LTD50  :{0}ms        LTD90  :{1}ms\n'.format(LTD[idx50], LTD[idx90]))
            w.write('AvgTD50:{0}ms        AvgTD90:{1}ms\n'.format(AvgTD[idx50], AvgTD[idx90]))


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
relative = False # just for conformer

train_wav_path = "egs/aishell/data/train/wav.scp"
train_text_path = "egs/aishell/data/train/text"
test_wav_path = "egs/aishell/data/test/wav.scp"
test_text_path = "egs/aishell/data/test/text"
vab_path = "egs/aishell/data/vocab"
latency_wav_path="Eval_latency/test/wav.scp"
latency_text_path="Eval_latency/test/text"
pinyin_dic = "Eval_latency/out"

batch_size = 16
train_epochs = 80
accum_steps = 4

ngpu = 1 if torch.cuda.is_available() else 0
print("ngpu: ", ngpu)

model = StreamingModel(fbank, channel, kernel_size, stride, vocab_size, d_model, n_heads, d_ff, enclayers, declayers, pre_norm, relative, windows=50)

solver = Solver(model, train_wav_path,train_text_path, test_wav_path, test_text_path,
                vab_path, fbank, batch_size, ngpu, train_epochs = train_epochs, accum_steps=accum_steps,
                 pinyin_dic=pinyin_dic, latency_wav_path=latency_wav_path, latency_text_path=latency_text_path)

solver.train()
solver.load_model("./result/streaming/streaming.pth")
solver.recognize()
solver.eval_latency()
