import torch
import torch.nn.functional as F
from .textdataset import TextDataset
from .vab import EOS, PAD, BOS
from torch.utils.data import DataLoader

def text_collate_fn(batch):
    utt_ids = [data[0] for data in batch]
    src_length = [len(data[1]) for data in batch]
    tgt_length = [len(data[2]) for data in batch]

    max_src_length = max(src_length)
    max_tgt_length = max(tgt_length) 

    padded_src = []
    padded_tgt = []
    padded_source_mask = []
    padded_target_mask = []


    for _, src_seq, tgt_seq in batch:
        padded_source_len = max_src_length - len(src_seq)
        padded_src.append([BOS] + src_seq + [PAD] * padded_source_len)
        padded_source_mask.append([1] * (len(src_seq) + 1) + [0] * padded_source_len)

        padded_target_len = max_tgt_length - len(tgt_seq)
        padded_tgt.append(tgt_seq + [EOS] + [PAD] * padded_target_len)
        padded_target_mask.append([1] * (len(tgt_seq) + 1) + [0] * padded_target_len)


    src_seqs = torch.LongTensor(padded_src)
    src_mask = torch.IntTensor(padded_source_mask) > 0
    tgt_seqs = torch.LongTensor(padded_tgt)
    tgt_mask = torch.IntTensor(padded_target_mask) > 0

    inputs = {
        'inputs': src_seqs,
        'mask': src_mask,
    }

    targets = {
        'targets': tgt_seqs,
        'mask': tgt_mask
    }
    return utt_ids, inputs, targets


class TextLoader(object):
    def __init__(self, text_path, vab_path, ngpu=1, batch_size=16):
        self.ngpu = ngpu
        self.batch_size = batch_size
        self.shuffle = True
        self.dataset = TextDataset(text_path, vab_path)
        self.batch_size *= ngpu

        self.loader = DataLoader(self.dataset, self.batch_size, 
                                 shuffle=self.shuffle,
                                 collate_fn=text_collate_fn)