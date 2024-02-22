import numpy as np
import torch
import time
import glob
import os

def from_bin_to_np(bin, length, expected_size):
    assert np.prod(expected_size) == length
    if bin > 0: assert np.log2(bin) <= length
    res = np.zeros((1, length), dtype=np.int8)
    i = 0
    while bin > 0:
        if bin & 1:
            res[:,i] = 1
        bin >>= 1
        i+=1
    return res.reshape(expected_size)

def fill_defaults(orig: dict, default: dict):
    for k in default.keys():
        if k not in orig:
            orig[k] = default[k]

def current_milli_time():
    return round(time.time() * 1000)

def find_files_like(folder, like):
    dd = [(filename, filename.split("_")[1]) for filename in [os.path.join(f) for f in glob.glob(os.path.join(folder, like + "*"))]]
    dd.sort(key=lambda x: int(x[1]))
    return dd

def find_latest(folder, like):
    dd = find_files_like(folder, like)
    return dd[-1][0] if len(dd) > 0 else None

def remove_if_max(folder, like, max_files):
    dd = find_files_like(folder, like)
    if len(dd) >= max_files: remove_file(dd[0][0])

def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)

def scale_torch_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    return tensor * scale + tensor.detach() * (1 - scale)

