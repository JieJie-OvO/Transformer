def load_vocab(path):
    unit2idx = {}
    with open(path, 'r', encoding='utf-8') as v:
        for line in v:
            unit, idx = line.strip().split()
            unit2idx[unit] = int(idx)
    return unit2idx

def get_idx2unit(unit2idx):
    idx2unit = {}
    for k,v in unit2idx.items():
        idx2unit[v] = k
    
    return idx2unit

def gen_characterlist(path, unit2idx, sosid=0, unkid=1):
    res = []
    with open(path, 'r', encoding='utf-8') as f:    
        for line in f:
            t = []
            t.append(sosid)
            parts = line.strip().split()
            idx = parts[0]
            phones = parts[1:]

            for ps in phones:
                for p in ps:
                    if p in unit2idx:
                        t.append(unit2idx[p])
                    else:
                        t.append(unkid)
        
            res.append(t)
    return res