from typing import Iterable, List

import torch



def sample_ar1_noise_from_sigma_seq(sigma_seq: torch.Tensor, rho: float) -> torch.Tensor:
    """Sample one AR(1) sequence with time-varying innovation scale.

    Uses the same convention as the uploaded data generator:
        e_0 = sigma_0 * z_0
        e_t = rho * e_{t-1} + sqrt(1-rho^2) * sigma_t * z_t
    """
    sigma_seq = sigma_seq.reshape(-1, 1)
    eps = torch.randn_like(sigma_seq)
    noise = torch.zeros_like(sigma_seq)
    noise[0] = sigma_seq[0] * eps[0]
    scale = torch.sqrt(torch.tensor(max(1.0 - rho ** 2, 1e-8), dtype=sigma_seq.dtype, device=sigma_seq.device))
    for k in range(1, sigma_seq.shape[0]):
        noise[k] = rho * noise[k - 1] + scale * sigma_seq[k] * eps[k]
    return noise



def sample_ar1_noise_from_sigma_batch(sigma_batch: torch.Tensor, rho: float) -> torch.Tensor:
    """Sample a batch of AR(1) sequences.

    sigma_batch shape: (B, T, 1) or (B, T)
    """
    if sigma_batch.ndim == 2:
        sigma_batch = sigma_batch.unsqueeze(-1)
    out = []
    for b in range(sigma_batch.shape[0]):
        out.append(sample_ar1_noise_from_sigma_seq(sigma_batch[b], rho))
    return torch.stack(out, dim=0)



def ar1_gaussian_nll_fixed_sigma(resid_all: torch.Tensor, sigma_all: torch.Tensor, rho: float, eps: float = 1e-6, reduce: str = "mean") -> torch.Tensor:
    """AR(1) Gaussian NLL with externally provided sigma and rho.

    resid_all: (B, T, 1) or (B, T)
    sigma_all: same shape as resid_all
    """
    if resid_all.ndim == 2:
        resid_all = resid_all.unsqueeze(-1)
    if sigma_all.ndim == 2:
        sigma_all = sigma_all.unsqueeze(-1)

    losses = []
    rho_t = torch.tensor(rho, dtype=resid_all.dtype, device=resid_all.device)
    rho_t = torch.clamp(rho_t, -0.999, 0.999)
    one_minus_rho2 = torch.clamp(1.0 - rho_t ** 2, min=eps)

    for b in range(resid_all.shape[0]):
        resid = resid_all[b].reshape(-1)
        sigma = sigma_all[b].reshape(-1).clamp_min(eps)
        sigma2 = sigma ** 2

        term0 = torch.log(sigma2[0]) + resid[0] ** 2 / sigma2[0]
        innov = resid[1:] - rho_t * resid[:-1]
        var_innov = one_minus_rho2 * sigma2[1:]
        terms_rest = torch.log(var_innov) + innov ** 2 / var_innov
        terms = torch.cat([term0.view(1), terms_rest], dim=0)
        losses.append(0.5 * terms.mean())

    losses = torch.stack(losses)
    if reduce == "mean":
        return losses.mean()
    if reduce == "sum":
        return losses.sum()
    raise ValueError("reduce must be 'mean' or 'sum'")



def gaussian_nll_fixed_sigma(resid: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6, reduce: str = "mean") -> torch.Tensor:
    resid = resid.reshape(-1)
    sigma = sigma.reshape(-1).clamp_min(eps)
    terms = torch.log(sigma ** 2) + (resid ** 2) / (sigma ** 2)
    if reduce == "mean":
        return 0.5 * terms.mean()
    if reduce == "sum":
        return 0.5 * terms.sum()
    raise ValueError("reduce must be 'mean' or 'sum'")



def parameter_anchor_penalty(modules: Iterable[torch.nn.Module], anchors: List[List[torch.Tensor]], prior_std: float = 1.0) -> torch.Tensor:
    total = 0.0
    prior_var = float(prior_std) ** 2
    for module, module_anchors in zip(modules, anchors):
        for param, anchor in zip(module.parameters(), module_anchors):
            total = total + torch.sum((param - anchor) ** 2) / prior_var
    return 0.5 * total
