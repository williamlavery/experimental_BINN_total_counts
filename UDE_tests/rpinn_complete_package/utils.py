import numpy as np
import torch


def set_seed(torch_seed=0, numpy_seed=0):
    torch.manual_seed(torch_seed)
    np.random.seed(numpy_seed)


def mean_and_ci(arr, ci=1.96):
    arr = np.array(arr)
    mean = arr.mean(axis=0)
    if arr.shape[0] == 1:
        lo = mean.copy()
        hi = mean.copy()
    else:
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        lo = mean - ci * sem
        hi = mean + ci * sem
    return mean, lo, hi
