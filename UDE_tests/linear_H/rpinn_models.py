import copy
from typing import Iterable, List

import torch
import torch.nn as nn


class SigmaEnsembleMean(nn.Module):
    """Frozen ensemble-mean sigma model.

    Wraps a list of pretrained sigma networks and returns the pointwise mean
    prediction. This is convenient when the learned data/physics uncertainty was
    fit repeatedly and you want a single frozen scale function for rPINNs.
    """

    def __init__(self, models: Iterable[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList([copy.deepcopy(m).eval() for m in models])
        for model in self.models:
            for p in model.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = [m(x) for m in self.models]
        return torch.stack(preds, dim=0).mean(dim=0)



def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module



def maybe_to_device(module: nn.Module, device: torch.device) -> nn.Module:
    module = module.to(device)
    return freeze_module(module)



def posterior_mean_and_ci(arr, ci: float = 1.96):
    arr = torch.as_tensor(arr)
    mean = arr.mean(dim=0)
    if arr.shape[0] == 1:
        return mean, mean.clone(), mean.clone()
    sem = arr.std(dim=0, unbiased=True) / (arr.shape[0] ** 0.5)
    lo = mean - ci * sem
    hi = mean + ci * sem
    return mean, lo, hi
