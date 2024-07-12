from .vab import load_vocab, UNK_TOKEN, EOS, BOS, PAD
from torch.utils.data import Dataset
from .augment import spec_augment
import numpy as np
import torchaudio as ta
import torch
import random

def normalization(feature):
    std, mean = torch.std_mean(feature)
    return (feature - mean) / std

class AudioDataset(Dataset):
    def __init__(self, wav_path, text_path, vab_path, fbank = 40, spec_augment = True):

        self.apply_spec_augment = spec_augment
        self.fbank = fbank
        self.unit2idx = load_vocab(vab_path)

        self.targets_dict = {}
        with open(text_path, 'r', encoding='utf-8') as t:
            for line in t:
                parts = line.strip().split()
                utt_id = parts[0]
                label = []
                for cs in parts[1:]:
                    for c in cs:
                        label.append(self.unit2idx[c] if c in self.unit2idx else self.unit2idx[UNK_TOKEN])
                self.targets_dict[utt_id] = label

        self.file_list = []
        with open(wav_path, 'r', encoding='utf-8') as fid:
            for line in fid:
                idx, path = line.strip().split()
                path = "./egs/aishell/"+path
                self.file_list.append([idx, path])

    def __getitem__(self, index):
        utt_id, path = self.file_list[index]
        wavform, sample_frequency = ta.load(path)

        feature = ta.compliance.kaldi.fbank(
            wavform, num_mel_bins= self.fbank,
            sample_frequency=sample_frequency, dither=0.0
            )

        feature = normalization(feature)

        if self.apply_spec_augment:
            feature = spec_augment(feature)

        feature_length = feature.shape[0]
        targets = self.targets_dict[utt_id]
        targets_length = len(targets)

        return utt_id, feature, feature_length, targets, targets_length

    def __len__(self):
        return len(self.file_list)

    def idx2unit(self):
        return {i: c for (c, i) in self.unit2idx.items()}

    def vocab_size(self):
        return len(self.unit2idx)
