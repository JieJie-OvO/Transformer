def compute_frames(total, i):
    a = 1
    b = 1
    if total%2 == 0:
        a = 0
        total = (total-2)/2
    else:
        a = 1
        total = (total-1)/2

    if total%2 == 0:
        b = 0
        total = (total-2)/2
    else:
        b = 1
        total = (total-1)/2
    
    if b == 1:
        i = i*2 +1
    else:
        i = i*2 +2
    
    if a == 1:
        i = i*2 +1
    else:
        i = i*2 +2
    
    return i