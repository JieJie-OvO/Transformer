import torch
import torch.nn.functional as F
from .dataset import AudioDataset
from .vab import EOS, PAD, BOS
from torch.utils.data import DataLoader


def collate_fn_with_eos_bos(batch):

    utt_ids = [data[0] for data in batch]
    features_length = [data[2] for data in batch]
    targets_length = [data[4] for data in batch]
    max_feature_length = max(features_length)
    max_target_length = max(targets_length)

    padded_features = []
    padded_targets = []
    padded_feature_mask = []
    padded_target_mask = []

    for _, feat, feat_len, target, target_len in batch:
        padding_feature_len = max_feature_length - feat_len
        padded_features.append(F.pad(feat, pad=(0, 0, 0, padding_feature_len), value=0.0).unsqueeze(0))
        padded_feature_mask.append([1] * feat_len + [0] * padding_feature_len)

        padding_target_len = max_target_length - target_len
        padded_targets.append([BOS] + target + [EOS] + [PAD] * padding_target_len)
        padded_target_mask.append([1] * (target_len + 2) + [0] * padding_target_len)

    features = torch.cat(padded_features, dim=0)
    features_length = torch.IntTensor(features_length)
    feature_mask = torch.IntTensor(padded_feature_mask) > 0

    targets = torch.LongTensor(padded_targets)
    targets_length = torch.IntTensor(targets_length).add(1)
    targets_mask = torch.IntTensor(padded_target_mask) > 0

    inputs = {
        'inputs': features,
        'inputs_length': features_length,
        'mask': feature_mask
    }

    targets = {
        'targets': targets,
        'targets_length': targets_length, # include eos
        'mask': targets_mask
    }

    return utt_ids, inputs, targets


class FeatureLoader(object):
    def __init__(self, wav_path, text_path, vab_path, fbank=40, spec_augment = True, ngpu=1, batch_size=16):
        self.ngpu = ngpu
        self.batch_size = batch_size
        self.shuffle = True if spec_augment else False
        self.dataset = AudioDataset(wav_path,text_path, vab_path, fbank, spec_augment=spec_augment)
        self.batch_size *= ngpu

        self.loader = DataLoader(self.dataset, self.batch_size, 
                                 shuffle=self.shuffle,
                                 collate_fn=collate_fn_with_eos_bos)
    