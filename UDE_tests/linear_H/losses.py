import torch


def gaussian_sigma_nll(resid, sigma, eps=1e-6, reduce="mean"):
    resid = resid.reshape(-1)
    sigma = sigma.reshape(-1).clamp_min(eps)
    terms = torch.log(sigma**2) + (resid**2) / (sigma**2)

    if reduce == "sum":
        return 0.5 * torch.sum(terms)
    elif reduce == "mean":
        return 0.5 * torch.mean(terms)
    raise ValueError("reduce must be 'mean' or 'sum'")


def ar1_gaussian_nll(resid, sigma, rho, eps=1e-6, reduce="mean"):
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
    elif reduce == "mean":
        return 0.5 * torch.mean(terms)
    raise ValueError("reduce must be 'mean' or 'sum'")


def ar1_gaussian_nll_batch(resid_all, sigma_all, rho, eps=1e-6, reduce="mean"):
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
    elif reduce == "sum":
        return losses.sum()
    raise ValueError("reduce must be 'mean' or 'sum'")


def pinn_physics_residual(sol_net, dyn_net, t_col):
    N_pred_col = sol_net(t_col)
    dN_dt = torch.autograd.grad(
        outputs=N_pred_col,
        inputs=t_col,
        grad_outputs=torch.ones_like(N_pred_col),
        create_graph=True,
        retain_graph=True,
    )[0]
    H_pred = dyn_net(N_pred_col)
    rhs_pred = N_pred_col * H_pred
    phys_res = dN_dt - rhs_pred
    return N_pred_col, dN_dt, rhs_pred, phys_res


def h_zero_nonnegative_penalty(dyn_net, device):
    N_zero = torch.zeros((1, 1), dtype=torch.float32, device=device)
    H_zero = dyn_net(N_zero)
    return torch.sum(torch.relu(-H_zero) ** 2)


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
