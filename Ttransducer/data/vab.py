BLK = 0
PAD = 0
BOS = 1
EOS = 1
UNK = 2
MASK = 3

BOS_TOKEN = '<S/E>'
EOS_TOKEN = '<S/E>'
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<unk>'
SPACE_TOKEN = '<SPACE>'
MASK_TOKEN = '<MASK>'

def load_vocab(path):
    unit2idx = {}
    with open(path, 'r', encoding='utf-8') as v:
        for line in v:
            unit, idx = line.strip().split()
            unit2idx[unit] = int(idx)
    return unit2idx