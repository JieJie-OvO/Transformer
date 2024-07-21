import torch
import torch.nn as nn
from .vocab_dict import load_vocab,get_idx2unit,gen_characterlist

class Ngram(nn.Module):
    def __init__(self, unit2idx_path, sentence_path, vocabsize=4232, n_gram=5):
        super(Ngram,self).__init__()
        self.unit2idx = load_vocab(unit2idx_path)
        self.idx2unit = get_idx2unit(self.unit2idx)
        self.vocabsize = vocabsize
        self.sentence_list = gen_characterlist(sentence_path, self.unit2idx)

        self.grams = []
        for i in range(n_gram):
            gram = self.gen_gram_p(n=i+1)
            self.grams.append(gram)
    
    def gen_gram_dict(self, sentence_list=None, n=1):
        '''
        gram_dict: str -> countdict
        count_dict: id -> count
        '''
        if sentence_list is None:
            sentence_list = self.sentence_list
        res = {}
        for s in sentence_list:
            for i in range(len(s)-n):
                subid = s[i:i+n]
                nextid = s[i+n]
                str = ""
                for id in subid:
                    str += self.idx2unit[id]
                if str not in res:
                    res[str] = {}
                if nextid not in res[str]:
                    res[str][nextid] = 0
                res[str][nextid] += 1
        return res
    
    def gen_gram_p(self, sentence_list=None, n=1):
        gram_dict = self.gen_gram_dict(sentence_list, n)

        res = {}
        for str, count_dict in gram_dict.items():
            res[str] = {}
            total = 0
            for k,v in count_dict.items():
                total += v
            if total == 0:
                total = 1
            for k,v in count_dict.items():
                res[str][k] = v/total
        
        return res

    def forward(self, x, n=5):
        x_len = len(x)
        if x_len < 5:
            n = x_len
        
        gram = self.grams[n-1]

        idlist = x[len(x)-n:]
        str=""
        for id in idlist:
           str += self.idx2unit[id]

        while str not in gram and n > 0:
            n = n-1
            gram = self.grams[n-1]
            idlist = x[len(x)-n:]
            str=""
            for id in idlist:
                str += self.idx2unit[id]

        predict = torch.zeros(self.vocabsize)
        if str in gram:
            str_pred = gram[str]
            for k,v in str_pred.items():
                predict[k] = v
        return predict