import numpy as np

def from_bin_to_np(bin, length):
    if bin > 0: assert np.log2(bin) <= length
    res = np.zeros((1, length), dtype=np.int8)
    i = 0
    while bin > 0:
        if bin & 1:
            res[:,i] = 1
        bin >>= 1
        i+=1
    return res