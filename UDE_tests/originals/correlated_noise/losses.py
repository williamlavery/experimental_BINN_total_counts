import numpy as np
import torch


import copy
import numpy as np
import torch



def time_derivative(model, t):
    """
    Compute du/dt for a scalar-output network u(t).
    """
    t_req = t.clone().detach().requires_grad_(True)
    u = model(t_req)
    du_dt = torch.autograd.grad(
        outputs=u,
        inputs=t_req,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    return u, du_dt


def pinn_residual(solution_net, dynamics_net, t_col):
    """
    Compute PINN residual:
        du/dt - G(u)
    """
    u_col, du_dt_col = time_derivative(solution_net, t_col)
    G_u = dynamics_net(u_col)
    return du_dt_col - G_u


def data_fit_loss(solution_net, t_obs, y_obs):
    pred = solution_net(t_obs)
    return torch.mean((pred - y_obs) ** 2)


def ic_loss(solution_net, t0, y0):
    pred0 = solution_net(t0)
    return torch.mean((pred0 - y0) ** 2)


def physics_loss(solution_net, dynamics_net, t_col):
    r = pinn_residual(solution_net, dynamics_net, t_col)
    return torch.mean(r ** 2)


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


def g_zero_nonnegative_penalty(dyn_net, device):
    N_zero = torch.zeros((1, 1), dtype=torch.float32, device=device)
    G_zero = dyn_net(N_zero)
    return torch.sum(torch.relu(-G_zero) ** 2)


def sigma_monotonicity_loss(model, N_col):
    N_req = N_col.clone().detach().requires_grad_(True)
    sigma_col = model(N_req)
    dsigma_dN = torch.autograd.grad(
        outputs=sigma_col,
        inputs=N_req,
        grad_outputs=torch.ones_like(sigma_col),
        create_graph=True,
        retain_graph=True,
    )[0]
    return torch.sum(torch.relu(-dsigma_dN) ** 2)


def ar1_gaussian_nll(resid, sigma, rho, eps=1e-6, reduce="mean"):
    """
    resid : [T, 1] or [T]
    sigma : [T, 1] or [T]
    rho   : scalar tensor or float with |rho| < 1
    """
    resid = resid.reshape(-1)
    sigma = sigma.reshape(-1).clamp_min(eps)

    if not torch.is_tensor(rho):
        rho = torch.tensor(rho, dtype=resid.dtype, device=resid.device)

    rho = torch.clamp(rho, -0.999, 0.999)

    sigma2 = sigma ** 2
    one_minus_rho2 = torch.clamp(1.0 - rho ** 2, min=eps)

    term0 = torch.log(sigma2[0]) + resid[0] ** 2 / sigma2[0]

    innov = resid[1:] - rho * resid[:-1]
    var_innov = one_minus_rho2 * sigma2[1:]
    terms_rest = torch.log(var_innov) + innov ** 2 / var_innov

    terms = torch.cat([term0.view(1), terms_rest], dim=0)

    if reduce == "sum":
        return 0.5 * torch.sum(terms)
    if reduce == "mean":
        return 0.5 * torch.mean(terms)
    raise ValueError("reduce must be 'mean' or 'sum'")


def ar1_gaussian_nll_batch(resid_all, sigma_all, rho, eps=1e-6, reduce="mean"):
    """
    resid_all : [B, T, 1] or [B, T]
    sigma_all : [B, T, 1] or [B, T]
    """
    if resid_all.ndim == 2:
        resid_all = resid_all.unsqueeze(-1)
    if sigma_all.ndim == 2:
        sigma_all = sigma_all.unsqueeze(-1)

    losses = []
    for b in range(resid_all.shape[0]):
        losses.append(
            ar1_gaussian_nll(
                resid_all[b], sigma_all[b], rho=rho, eps=eps, reduce="mean"
            )
        )
    losses = torch.stack(losses)

    if reduce == "mean":
        return losses.mean()
    if reduce == "sum":
        return losses.sum()
    raise ValueError("reduce must be 'mean' or 'sum'")