import torch

'''
1 0 0
1 1 0
1 1 1
'''
def get_tril_mask(targets):
    if targets.dim() == 2:
        batch_size, steps = targets.size()
    else:
        batch_size, steps, _ = targets.size()
    seq_mask = torch.ones([batch_size, steps, steps], device=targets.device)
    seq_mask = torch.tril(seq_mask).bool()
    return seq_mask

'''
chunk_size = 2
1 1  0 0  0 0  0
1 1  0 0  0 0  0
1 1  1 1  0 0  0
1 1  1 1  0 0  0
0 0  1 1  1 1  0
0 0  1 1  1 1  0
0 0  0 0  1 1  1
'''
def get_chunk_lookahead_mask(x, chunk_size):
    batch_size, steps, _ = x.size()
    mask = torch.zeros([batch_size, steps, steps], device=x.device)
    start = 0
    last = chunk_size
    for i in range(0, steps):
        if i >= last:
            start += chunk_size
            last += chunk_size
        begin = start if start-chunk_size < 0 else start-chunk_size
        end = last if last < steps else steps
        mask[:, i, begin:end] = 1
    mask = mask.bool()
    return mask

'''
chunk_size = 2
1 1  0 0  0 0  0
1 1  0 0  0 0  0
0 0  1 1  0 0  0
0 0  1 1  0 0  0
0 0  0 0  1 1  0
0 0  0 0  1 1  0
0 0  0 0  0 0  1
'''
def get_chunk_mask(x, chunk_size):
    batch_size, steps, _ = x.size()
    mask = torch.zeros([batch_size, steps, steps], device=x.device)
    start = 0
    last = chunk_size
    for i in range(0, steps):
        if i >= last:
            start += chunk_size
            last += chunk_size
        begin = start
        end = last if last < steps else steps
        mask[:, i, begin:end] = 1
    mask = mask.bool()
    return mask