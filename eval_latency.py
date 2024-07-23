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
        reses = textgrid2outlist(self.pinyin_dic)
        self.model.eval()
        accu_time = 0
        total_frames = 0

        latencys = []

        for step, (utt_id, inputs, targets) in enumerate(self.test_loader.loader):
            latency = []
            print( reses[utt_id])
            exit()
            if step == 0:
                # 最开始跳过，系统还没有稳定
                continue
            
            if self.ngpu>0:
                inputs = map_to_cuda(inputs)
            total_frames = inputs['inputs'].size(1)

            res = reses[utt_id]
        
            st = time.time()
            preds = self.model.recognize_with_latency(inputs)

            pred = preds[0]
            for i in range(len(pred)):
                # [[now, t], pred]
                idx = compute_frames(total_frames, pred[i][0][1])
                et = pred[i][0][0]
                delta = idx * 0.01 + et - st
                min = delta - res[i][1]
                max = delta - res[i][0]
                latency.append([min, max])
                    
            latencys.append(latency)

        # TODO计算结果


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
test_wav_path = "eval_latency/test/wav.scp"
test_text_path = "eval_latency/test/text"
vab_path = "egs/aishell/data/transducer_vab"

pinyin_dic = "eval_latency/out"

batch_size = 10
train_epochs = 80
accum_steps = 6

ngpu = 1 if torch.cuda.is_available() else 0
print("ngpu: ", ngpu)

model = Transducer(fbank, input_size, enc_hidden, enc_out, enc_layers, 
                   dec_hidden, vocab_size, dec_out, dec_layers, 
                   joint_dim, 
                   dropout, predict_strategy=predict_strategy, lmweight=lmweight)

solver = Eval_latency(model, train_wav_path,train_text_path, test_wav_path, test_text_path,
                vab_path, fbank, batch_size, ngpu, pinyin_dic=pinyin_dic)

solver.load_model("./pth/model.epoch.4.pth")
solver.recognize()
