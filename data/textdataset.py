import torch
import logging
from .vab import load_vocab
from torch.utils.data import Dataset



class TextDataset(Dataset):
    def __init__(self, text_path, vab_path):
        self.src_unit2idx = load_vocab(vab_path)
        self.tgt_unit2idx = load_vocab(vab_path)

        self.src_list = []
        self.tgt_dict = {}
        with open(text_path, 'r', encoding='utf-8') as t:
            for line in t:
                parts = line.strip().split()
                utt_id = parts[0]
                label = []
                for cs in parts[1:]:
                    for c in cs:
                        label.append(self.src_unit2idx[c] if c in self.src_unit2idx else self.src_unit2idx[UNK_TOKEN])
                self.src_list.append((utt_id, label))

        with open(text_path, 'r', encoding='utf-8') as t:
            for line in t:
                parts = line.strip().split()
                utt_id = parts[0]
                label = []
                for cs in parts[1:]:
                    for c in cs:
                        label.append(self.tgt_unit2idx[c] if c in self.tgt_unit2idx else self.tgt_unit2idx[UNK_TOKEN])
                self.tgt_dict[utt_id] = label

        assert len(self.src_list) == len(self.tgt_dict)

        self.lengths = len(self.src_list)

    def __getitem__(self, index):
        idx, src_seq = self.src_list[index]
        tgt_seq = self.tgt_dict[idx]

        return idx, src_seq, tgt_seq

    def __len__(self):
        return self.lengths

    def src_vocab_size(self):
        return len(self.src_unit2idx)

    def tgt_vocab_size(self):
        return len(self.tgt_unit2idx)

    def src_idx2unit(self):
        return {i: c for (c, i) in self.src_unit2idx.items()}

    def tgt_idx2unit(self):
        return {i: c for (c, i) in self.tgt_unit2idx.items()}

