from Latency.model.transducer import Transducer
from Latency.data.dataloader import FeatureLoader
from Latency.train.utils import map_to_cuda
import editdistance
import torch
import time
from Eval_latency.read_textgrid import textgrid2outlist
from Eval_latency.tools import compute_frames


class Eval_latency():
    def __init__(self, model, train_wav_path, train_text_path, test_wav_path, 
                 test_text_path, vab_path, fbank = 40, batch_size=16, ngpu=1,
                 pinyin_dic = "eval_latency/out"):
        self.model = model
        self.ngpu = ngpu
        self.pinyin_dic = pinyin_dic
        if ngpu >= 1:
            model.cuda()
        
        self.train_loader = FeatureLoader(train_wav_path, train_text_path, vab_path, fbank, spec_augment=True, ngpu=1, batch_size=batch_size)
        self.test_loader = FeatureLoader(test_wav_path, test_text_path, vab_path, fbank, spec_augment=False, ngpu=1, batch_size=1)

    def load_model(self, path):
        chkpt = torch.load(path)
        self.model.load_model(chkpt)

    def recognize(self):
        idx2unit = self.train_loader.dataset.idx2unit()
        reses = textgrid2outlist(self.pinyin_dic)
        self.model.eval()
        accu_time = 0
        total_frames = 0

        latencys = []

        for step, (utt_id, inputs, targets) in enumerate(self.test_loader.loader):
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

        # TODO计算结果
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
fbank=80
input_size=80
enc_hidden=512
enc_out=340
enc_layers=6
dec_hidden=512
vocab_size=4232
dec_out=320
dec_layers=1
joint_dim=512
dropout = 0.2
predict_strategy="RNN-T"
lmweight = 0.2

train_wav_path = "egs/aishell/data/train/wav.scp"
train_text_path = "egs/aishell/data/train/text"
test_wav_path = "Eval_latency/test/wav.scp"
test_text_path = "Eval_latency/test/text"
vab_path = "egs/aishell/data/transducer_vab"

pinyin_dic = "Eval_latency/out"

batch_size = 10
train_epochs = 80
accum_steps = 6

ngpu = 0 if torch.cuda.is_available() else 0
print("ngpu: ", ngpu)

model = Transducer(fbank, input_size, enc_hidden, enc_out, enc_layers, 
                   dec_hidden, vocab_size, dec_out, dec_layers, 
                   joint_dim, 
                   dropout, predict_strategy=predict_strategy, lmweight=lmweight)

solver = Eval_latency(model, train_wav_path,train_text_path, test_wav_path, test_text_path,
                vab_path, fbank, batch_size, ngpu, pinyin_dic=pinyin_dic)

solver.load_model("./result/latency/rnnt.pth")
solver.recognize()
