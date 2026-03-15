import copy
import numpy as np
import torch
import torch.nn as nn

from data import odeint_rk4
from losses import (
    ar1_gaussian_nll_batch,
    pinn_physics_residual,
    randomized_residual_nll,
)
from models import DynamicsNet, SigmaNet, SolutionNet
from utils import mean_and_ci


@torch.no_grad()
def rollout_with_learned_rhs(dyn_net, y0, t):
    y_roll = odeint_rk4(lambda tt, yy: dyn_net(yy), y0, t.squeeze()).squeeze(1)
    return y_roll


def _ensure_2d_col(x, device=None, dtype=None):
    if not torch.is_tensor(x):
        x = torch.tensor(x, device=device, dtype=dtype)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    return x


def _expand_time_like_state(t_like, N_like):
    """
    Broadcast time tensor so it matches N_like shape.
    Accepted inputs:
      - t_like shape [T, 1]
      - t_like shape [B, T, 1]
      - t_like shape matching N_like
    """
    if t_like.shape == N_like.shape:
        return t_like

    if t_like.ndim == 2 and N_like.ndim == 3:
        return t_like.unsqueeze(0).expand(N_like.shape[0], -1, -1)

    if t_like.ndim == 1 and N_like.ndim == 3:
        return t_like.view(1, -1, 1).expand(N_like.shape[0], -1, -1)

    if t_like.ndim == 1 and N_like.ndim == 2:
        return t_like.view(-1, 1)

    raise ValueError(
        f"Could not broadcast t_like with shape {tuple(t_like.shape)} "
        f"to match N_like with shape {tuple(N_like.shape)}."
    )


def _make_sigma_features(N, t):
    """
    Build feature tensor for sigma_net from state N and time t.
    Returns shape [..., 2] with columns [N, t].
    """
    if t.shape != N.shape:
        t = _expand_time_like_state(t, N)
    return torch.cat([N, t], dim=-1)


def _sigma_monotonicity_loss_nt(
    sigma_net,
    N_col_base,
    t_col_base,
):
    """
    Penalize negative d sigma(N, t) / dN while holding t fixed.
    """
    N_in = N_col_base.clone().detach().requires_grad_(True)
    t_in = t_col_base.clone().detach()

    sigma_in = _make_sigma_features(N_in, t_in)
    sigma_pred = sigma_net(sigma_in)

    grad_N = torch.autograd.grad(
        outputs=sigma_pred.sum(),
        inputs=N_in,
        create_graph=True,
        retain_graph=True,
    )[0]

    return torch.relu(-grad_N).mean()


def _nn_weight_penalty(
    module_or_modules,
    p=2.0,
    include_bias=False,
    normalize=False,
):
    """
    Penalty over NN weights only by default.

    Args:
        module_or_modules: nn.Module or iterable of nn.Module
        p: norm power. p=2 -> sum(w^2), p=1 -> sum(|w|)
        include_bias: whether to include bias parameters
        normalize: divide by number of penalized scalars
    """
    if module_or_modules is None:
        return torch.tensor(0.0)

    if isinstance(module_or_modules, nn.Module):
        modules = [module_or_modules]
    else:
        modules = list(module_or_modules)

    params = []
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if (not include_bias) and name.endswith("bias"):
                continue
            params.append(param)

    if len(params) == 0:
        ref = None
        for module in modules:
            for param in module.parameters():
                ref = param
                break
            if ref is not None:
                break
        if ref is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=ref.device, dtype=ref.dtype)

    if p == 2 or p == 2.0:
        penalty = sum((param ** 2).sum() for param in params)
    elif p == 1 or p == 1.0:
        penalty = sum(param.abs().sum() for param in params)
    else:
        penalty = sum(param.abs().pow(p).sum() for param in params)

    if normalize:
        denom = sum(param.numel() for param in params)
        penalty = penalty / max(denom, 1)

    return penalty




def ar1_gaussian_nll_markov(resid, sigma, rho, eps=1e-6, reduce="mean"):
    """
    Markov-consistent Gaussian AR(1) NLL where sigma_t is the marginal / total
    residual standard deviation at time t.

    Model:
        e_0 ~ N(0, sigma_0^2)

        e_t | e_{t-1} ~ N(
            rho * (sigma_t / sigma_{t-1}) * e_{t-1},
            (1 - rho^2) * sigma_t^2
        )

    This parameterization ensures that if Var(e_{t-1}) = sigma_{t-1}^2 then
    Var(e_t) = sigma_t^2.

    Args
    ----
    resid : tensor, shape [T] or [T, 1]
    sigma : tensor, shape [T] or [T, 1]
        Predicted marginal / total std at each time.
    rho : scalar tensor or float
    eps : float
    reduce : {"mean", "sum", "none"}

    Returns
    -------
    scalar tensor if reduce in {"mean", "sum"}, else tensor [T]
    """
    resid = resid.reshape(-1)
    sigma = sigma.reshape(-1).clamp_min(eps)

    if not torch.is_tensor(rho):
        rho = torch.tensor(rho, dtype=resid.dtype, device=resid.device)
    rho = torch.clamp(rho, -0.999, 0.999)

    sigma2 = sigma ** 2
    one_minus_rho2 = torch.clamp(1.0 - rho ** 2, min=eps)

    # Initial marginal density: e_0 ~ N(0, sigma_0^2)
    term0 = torch.log(sigma2[0]) + resid[0] ** 2 / sigma2[0]

    # Transition density:
    # e_t | e_{t-1} ~ N(rho * (sigma_t / sigma_{t-1}) * e_{t-1},
    #                  (1-rho^2) * sigma_t^2)
    scale_ratio = sigma[1:] / sigma[:-1].clamp_min(eps)
    cond_mean = rho * scale_ratio * resid[:-1]
    innov = resid[1:] - cond_mean
    var_innov = one_minus_rho2 * sigma2[1:]

    terms_rest = torch.log(var_innov) + innov ** 2 / var_innov
    terms = torch.cat([term0.view(1), terms_rest], dim=0)

    if reduce == "sum":
        return 0.5 * torch.sum(terms)
    elif reduce == "mean":
        return 0.5 * torch.mean(terms)
    elif reduce == "none":
        return 0.5 * terms
    raise ValueError("reduce must be 'mean', 'sum', or 'none'")


def ar1_gaussian_nll_markov_batch(resid_all, sigma_all, rho, eps=1e-6, reduce="mean"):
    """
    Batch version of ar1_gaussian_nll_markov.

    Args
    ----
    resid_all : [B, T, 1] or [B, T]
    sigma_all : [B, T, 1] or [B, T]
    rho : scalar tensor or float
    eps : float
    reduce : {"mean", "sum", "none"}

    Returns
    -------
    If reduce == "none": [B]
    Else scalar tensor.
    """
    if resid_all.ndim == 3:
        resid_all = resid_all.squeeze(-1)
    if sigma_all.ndim == 3:
        sigma_all = sigma_all.squeeze(-1)

    losses = []
    for b in range(resid_all.shape[0]):
        losses.append(
            ar1_gaussian_nll_markov(
                resid=resid_all[b],
                sigma=sigma_all[b],
                rho=rho,
                eps=eps,
                reduce="mean",
            )
        )
    losses = torch.stack(losses, dim=0)

    if reduce == "none":
        return losses
    elif reduce == "mean":
        return losses.mean()
    elif reduce == "sum":
        return losses.sum()
    raise ValueError("reduce must be 'mean', 'sum', or 'none'")


def ar1_markov_conditional_stats(resid, sigma, rho, eps=1e-6):
    """
    Compute conditional means and stds under the Markov-consistent AR(1) model.

    Returns
    -------
    cond_mean : [T]
        E[e_t | e_{t-1}] with cond_mean[0] = 0
    cond_std : [T]
        Std[e_t | e_{t-1}] with cond_std[0] = sigma_0
    """
    resid = resid.reshape(-1)
    sigma = sigma.reshape(-1).clamp_min(eps)

    if not torch.is_tensor(rho):
        rho = torch.tensor(rho, dtype=resid.dtype, device=resid.device)
    rho = torch.clamp(rho, -0.999, 0.999)

    T = resid.shape[0]
    cond_mean = torch.zeros_like(resid)
    cond_std = torch.zeros_like(sigma)

    cond_mean[0] = 0.0
    cond_std[0] = sigma[0]

    ratio = sigma[1:] / sigma[:-1].clamp_min(eps)
    cond_mean[1:] = rho * ratio * resid[:-1]
    cond_std[1:] = torch.sqrt(torch.clamp(1.0 - rho**2, min=eps)) * sigma[1:]

    return cond_mean, cond_std


def fit_sigma_ar1_repeated(
    N_seq_all,
    t_seq_all,
    resid_seq_all,
    N_eval_grid,
    t_eval_grid,
    N_col_base,
    t_col_base,
    device,
    n_repeats=5,
    val_fraction=0.2,
    n_epochs=3000,
    lr=5e-3,
    hidden_dim=8,
    lambda_mon=1.0,
    lambda_reg=1e-5,
    sigma_weight_decay=None,
    sigma_reg_p=2.0,
    sigma_reg_include_bias=False,
    sigma_reg_normalize=False,
    seed_offset_split=5000,
    seed_offset_model=7000,
    label="sigma-ar1-markov-nt",
    print_every=500,
    eps_nll=1e-6,
):
    """
    Fit sigma = sigma(N, t) using repeated train/validation splits under the
    Markov-consistent AR(1) noise model.

    IMPORTANT INTERPRETATION
    ------------------------
    sigma_net(N, t) outputs the MARGINAL / TOTAL residual standard deviation,
    not merely the innovation scale.

    Noise model:
        e_0 ~ N(0, sigma_0^2)
        e_t | e_{t-1} ~ N(
            rho * (sigma_t / sigma_{t-1}) * e_{t-1},
            (1 - rho^2) * sigma_t^2
        )

    So sigma_t is the full uncertainty at time t, while the one-step
    conditional uncertainty is sqrt(1-rho^2) * sigma_t.

    Expected shapes:
      N_seq_all     : [B, T, 1]
      t_seq_all     : [B, T, 1] or [T, 1]
      resid_seq_all : [B, T, 1]
      N_eval_grid   : [M, 1]
      t_eval_grid   : [M, 1]
      N_col_base    : [K, 1]
      t_col_base    : [K, 1]
    """
    if sigma_weight_decay is None:
        sigma_weight_decay = lambda_reg

    B = N_seq_all.shape[0]
    t_seq_all = _expand_time_like_state(t_seq_all, N_seq_all)

    use_validation = val_fraction > 0.0 and B > 1

    if use_validation:
        n_val = int(round(val_fraction * B))
        n_val = max(1, min(n_val, B - 1))
    else:
        n_val = 0
    n_train = B - n_val

    train_total_histories = []
    train_nll_histories = []
    train_mon_histories = []
    train_reg_histories = []

    val_total_histories = []
    val_nll_histories = []
    val_mon_histories = []
    val_reg_histories = []

    # sigma_curve_histories stores the marginal / total sigma curve
    sigma_curve_histories = []

    # innovation_curve_histories stores sqrt(1-rho^2) * sigma_curve
    innovation_curve_histories = []

    rho_histories = []

    best_val_losses = []
    best_epochs = []
    best_rhos = []
    best_models = []

    sigma_eval_features = _make_sigma_features(N_eval_grid, t_eval_grid)

    for repeat in range(n_repeats):
        g = torch.Generator(device=device)
        g.manual_seed(seed_offset_split + repeat)

        perm = torch.randperm(B, generator=g, device=device)
        train_idx = perm[:n_train].sort().values

        if use_validation:
            val_idx = perm[n_train:].sort().values
        else:
            val_idx = torch.empty(0, dtype=torch.long, device=device)

        N_train = N_seq_all[train_idx]
        t_train = t_seq_all[train_idx]
        r_train = resid_seq_all[train_idx]

        if use_validation:
            N_val = N_seq_all[val_idx]
            t_val = t_seq_all[val_idx]
            r_val = resid_seq_all[val_idx]

        torch.manual_seed(seed_offset_model + repeat)
        np.random.seed(seed_offset_model + repeat)

        sigma_net = SigmaNet(in_dim=2, hidden_dim=hidden_dim).to(device)
        rho_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))

        opt = torch.optim.Adam(list(sigma_net.parameters()) + [rho_raw], lr=lr)

        train_total_history = []
        train_nll_history = []
        train_mon_history = []
        train_reg_history = []

        val_total_history = []
        val_nll_history = []
        val_mon_history = []
        val_reg_history = []

        best_val_loss = float("inf")
        best_epoch = -1
        best_state = None

        for epoch in range(n_epochs):
            opt.zero_grad()

            rho = torch.tanh(rho_raw)

            sigma_pred_train = sigma_net(_make_sigma_features(N_train, t_train))

            train_nll = ar1_gaussian_nll_markov_batch(
                resid_all=r_train,
                sigma_all=sigma_pred_train,
                rho=rho,
                eps=eps_nll,
                reduce="mean",
            )

            train_mon = _sigma_monotonicity_loss_nt(
                sigma_net=sigma_net,
                N_col_base=N_col_base,
                t_col_base=t_col_base,
            )

            train_reg = sigma_weight_decay * _nn_weight_penalty(
                sigma_net,
                p=sigma_reg_p,
                include_bias=sigma_reg_include_bias,
                normalize=sigma_reg_normalize,
            )

            train_total = train_nll + lambda_mon * train_mon + train_reg
            train_total.backward()
            opt.step()

            rho_eval = torch.tanh(rho_raw).detach()

            if use_validation:
                with torch.no_grad():
                    sigma_pred_val = sigma_net(_make_sigma_features(N_val, t_val))
                    val_nll = ar1_gaussian_nll_markov_batch(
                        resid_all=r_val,
                        sigma_all=sigma_pred_val,
                        rho=rho_eval,
                        eps=eps_nll,
                        reduce="mean",
                    )

                val_mon = _sigma_monotonicity_loss_nt(
                    sigma_net=sigma_net,
                    N_col_base=N_col_base,
                    t_col_base=t_col_base,
                )

                val_reg = sigma_weight_decay * _nn_weight_penalty(
                    sigma_net,
                    p=sigma_reg_p,
                    include_bias=sigma_reg_include_bias,
                    normalize=sigma_reg_normalize,
                )

                val_total = val_nll + lambda_mon * val_mon + val_reg
                criterion = val_total.item()
            else:
                val_nll = torch.tensor(np.nan, device=device)
                val_mon = torch.tensor(np.nan, device=device)
                val_reg = torch.tensor(np.nan, device=device)
                val_total = torch.tensor(np.nan, device=device)
                criterion = train_total.item()

            train_total_history.append(train_total.item())
            train_nll_history.append(train_nll.item())
            train_mon_history.append(train_mon.item())
            train_reg_history.append(train_reg.item())

            val_total_history.append(val_total.item())
            val_nll_history.append(val_nll.item())
            val_mon_history.append(val_mon.item())
            val_reg_history.append(val_reg.item())

            if (epoch == 0) or ((epoch + 1) % print_every == 0) or (epoch == n_epochs - 1):
                print(
                    f"{label} | Repeat {repeat + 1}/{n_repeats} | "
                    f"Epoch {epoch + 1}/{n_epochs} | rho={rho_eval.item():.4f} | "
                    f"Train Total={train_total.item():.6e} | "
                    f"Train NLL={train_nll.item():.6e} | "
                    f"Train Reg={train_reg.item():.6e} | "
                    f"Val Total={val_total.item():.6e}"
                )

            if criterion < best_val_loss:
                best_val_loss = criterion
                best_epoch = epoch
                best_state = {
                    "sigma_net": copy.deepcopy(sigma_net.state_dict()),
                    "rho_raw": rho_raw.detach().clone(),
                }

        sigma_net.load_state_dict(best_state["sigma_net"])
        sigma_net.eval()

        rho_best = torch.tanh(best_state["rho_raw"]).item()

        with torch.no_grad():
            sigma_curve = sigma_net(sigma_eval_features).squeeze(1)
            innovation_curve = np.sqrt(max(1.0 - rho_best**2, eps_nll)) * sigma_curve
            sigma_curve_np = sigma_curve.cpu().numpy()
            innovation_curve_np = innovation_curve.cpu().numpy()

        best_models.append(copy.deepcopy(sigma_net))

        train_total_histories.append(train_total_history)
        train_nll_histories.append(train_nll_history)
        train_mon_histories.append(train_mon_history)
        train_reg_histories.append(train_reg_history)

        val_total_histories.append(val_total_history)
        val_nll_histories.append(val_nll_history)
        val_mon_histories.append(val_mon_history)
        val_reg_histories.append(val_reg_history)

        sigma_curve_histories.append(sigma_curve_np)
        innovation_curve_histories.append(innovation_curve_np)
        rho_histories.append(rho_best)

        best_val_losses.append(best_val_loss)
        best_epochs.append(best_epoch)
        best_rhos.append(rho_best)

    return {
        "train_total_histories": np.array(train_total_histories),
        "train_nll_histories": np.array(train_nll_histories),
        "train_mon_histories": np.array(train_mon_histories),
        "train_reg_histories": np.array(train_reg_histories),
        "val_total_histories": np.array(val_total_histories),
        "val_nll_histories": np.array(val_nll_histories),
        "val_mon_histories": np.array(val_mon_histories),
        "val_reg_histories": np.array(val_reg_histories),

        # Main quantity: marginal / total uncertainty sigma(N, t)
        "curve_histories": np.array(sigma_curve_histories),
        "sigma_total_histories": np.array(sigma_curve_histories),

        # One-step conditional std = sqrt(1-rho^2) * sigma(N, t)
        "innovation_curve_histories": np.array(innovation_curve_histories),

        "rho_histories": np.array(rho_histories),
        "best_val_losses": np.array(best_val_losses),
        "best_epochs": np.array(best_epochs),
        "best_rhos": np.array(best_rhos),
        "best_models": best_models,
    }

def _build_randomization_banks(cfg, t_obs, device, n_ics, sample_idx):
    rng = torch.Generator(device=device)
    rng.manual_seed(cfg.seed_torch + 100000 + sample_idx)

    data_xi = torch.randn(
        n_ics,
        t_obs.shape[0],
        1,
        generator=rng,
        device=device,
        dtype=t_obs.dtype,
    )

    t_col_bank = cfg.t0 + (cfg.t1 - cfg.t0) * torch.rand(
        n_ics,
        cfg.n_col,
        1,
        generator=rng,
        device=device,
        dtype=t_obs.dtype,
    )
    xi_phys_bank = torch.randn(
        n_ics,
        cfg.n_col,
        1,
        generator=rng,
        device=device,
        dtype=t_obs.dtype,
    )
    return data_xi, t_col_bank, xi_phys_bank


def _resolve_sigma_models(sigma_model=None, sigma_fit_result=None):
    if sigma_model is not None and sigma_fit_result is not None:
        raise ValueError("Provide either sigma_model or sigma_fit_result, not both.")

    source = sigma_model if sigma_model is not None else sigma_fit_result
    if source is None:
        return None

    if isinstance(source, dict):
        if "best_models" in source:
            return list(source["best_models"])
        if "sigma_model" in source:
            return [source["sigma_model"]]
        raise ValueError(
            "sigma_fit_result dict must contain 'best_models' or 'sigma_model'."
        )

    if isinstance(source, (list, tuple)):
        return list(source)

    return [source]


def _sigma_from_models(sigma_models, N, t, sigma_floor=1e-4):
    sigma_preds = []
    sigma_features = _make_sigma_features(N, t)
    for model in sigma_models:
        sigma_preds.append(model(sigma_features))
    sigma = torch.stack(sigma_preds, dim=0).mean(dim=0)
    return torch.clamp(sigma, min=sigma_floor)


def fit_multi_ic_rpinn(
    cfg,
    y_data_all,
    t_obs,
    t_plot,
    N_grid,
    n_ics,
    device,
    sigma_model=None,
    sigma_fit_result=None,
):
    train_pinn_total_histories = []
    train_data_histories = []
    train_phys_histories = []
    train_ic_histories = []
    train_optim_total_histories = []
    train_reg_histories = []

    traj_histories = []
    rhs_histories = []
    h_histories = []
    obs_fit_histories = []
    sample_states = []

    sigma_models = _resolve_sigma_models(
        sigma_model=sigma_model,
        sigma_fit_result=sigma_fit_result,
    )

    for model in sigma_models or []:
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    use_learned_sigma = bool(
        getattr(cfg, "use_learned_sigma_rpinn", True) and sigma_models is not None
    )
    sigma_eval_mode = getattr(cfg, "sigma_eval_mode", "predicted_state")
    sigma_detach_state = bool(getattr(cfg, "sigma_detach_state", True))
    sigma_floor = float(getattr(cfg, "sigma_floor_rpinn", 1e-4))

    sigma_data_const = float(cfg.sigma_data_rpinn)
    sigma_phys_const = float(cfg.sigma_phys_rpinn)

    # Weight penalty settings for the PINN networks.
    # Defaults keep old behavior when not set.
    lambda_dyn_reg = float(getattr(cfg, "lambda_dyn_reg", 0.0))
    lambda_sol_reg = float(getattr(cfg, "lambda_sol_reg", 0.0))
    nn_reg_p = float(getattr(cfg, "nn_reg_p", 2.0))
    nn_reg_include_bias = bool(getattr(cfg, "nn_reg_include_bias", False))
    nn_reg_normalize = bool(getattr(cfg, "nn_reg_normalize", False))

    def _sigma_for_data(N_pred_train, y_train, t_train):
        if not use_learned_sigma:
            return sigma_data_const

        if sigma_eval_mode == "predicted_state":
            N_sigma = N_pred_train
        elif sigma_eval_mode == "observed_state":
            N_sigma = y_train
        else:
            raise ValueError(
                "sigma_eval_mode must be 'predicted_state' or 'observed_state'"
            )

        if sigma_detach_state:
            N_sigma = N_sigma.detach()

        return _sigma_from_models(
            sigma_models,
            N=N_sigma,
            t=t_train.detach(),
            sigma_floor=sigma_floor,
        )

    def _sigma_for_physics(N_pred_col, t_col):
        if not use_learned_sigma:
            return sigma_phys_const

        N_sigma = N_pred_col.detach() if sigma_detach_state else N_pred_col
        return _sigma_from_models(
            sigma_models,
            N=N_sigma,
            t=t_col.detach(),
            sigma_floor=sigma_floor,
        )

    def _train_single_run(
        sample_idx,
        randomized,
        init_from_state=None,
        dyn_seed=None,
        sol_seed_base=None,
    ):
        if dyn_seed is None:
            dyn_seed = cfg.dyn_init_seed_base + sample_idx
        if sol_seed_base is None:
            sol_seed_base = cfg.sol_init_seed_base + 1000 * sample_idx

        torch.manual_seed(dyn_seed)
        np.random.seed(dyn_seed)
        dyn_net = DynamicsNet(
            hidden_dim=cfg.hidden_dim_dyn,
            factor_rhs_by_state=cfg.factor_rhs_by_state,
        ).to(device)

        sol_nets = []
        sol_init_seeds = []
        for ic_idx in range(n_ics):
            init_seed = sol_seed_base + ic_idx
            sol_init_seeds.append(init_seed)
            torch.manual_seed(init_seed)
            np.random.seed(init_seed)
            sol_nets.append(SolutionNet(hidden_dim=cfg.hidden_dim_sol).to(device))
        sol_nets = nn.ModuleList(sol_nets)

        if init_from_state is not None:
            dyn_net.load_state_dict(copy.deepcopy(init_from_state["dyn_net"]))
            for ic_idx in range(n_ics):
                sol_nets[ic_idx].load_state_dict(
                    copy.deepcopy(init_from_state["sol_nets"][ic_idx])
                )

        if randomized:
            data_xi_bank, t_col_bank, xi_phys_bank = _build_randomization_banks(
                cfg=cfg,
                t_obs=t_obs,
                device=device,
                n_ics=n_ics,
                sample_idx=sample_idx,
            )
        else:
            t_col_bank = []
            zero_data_xi_bank = []
            zero_phys_xi_bank = []
            for _ in range(n_ics):
                t_col_ic = torch.linspace(
                    float(t_obs.min()),
                    float(t_obs.max()),
                    cfg.n_col,
                    device=device,
                    dtype=t_obs.dtype,
                ).reshape(-1, 1)
                t_col_bank.append(t_col_ic)
                zero_data_xi_bank.append(torch.zeros_like(t_obs))
                zero_phys_xi_bank.append(
                    torch.zeros((cfg.n_col, 1), device=device, dtype=t_obs.dtype)
                )
            data_xi_bank = zero_data_xi_bank
            xi_phys_bank = zero_phys_xi_bank

        optimizer = torch.optim.Adam(
            list(dyn_net.parameters()) + list(sol_nets.parameters()),
            lr=cfg.learning_rate_pinn,
        )

        train_pinn_total_history = []
        train_data_history = []
        train_phys_history = []
        train_ic_history = []
        train_optim_total_history = []
        train_reg_history = []

        best_train_loss = float("inf")
        best_epoch = -1
        best_state = None

        for epoch in range(cfg.n_epochs_pinn):
            optimizer.zero_grad()

            total_data_loss = 0.0
            total_phys_loss = 0.0
            total_ic_loss = 0.0
            total_sigma_data = 0.0
            total_sigma_phys = 0.0

            for ic_idx in range(n_ics):
                sol_net = sol_nets[ic_idx]

                if cfg.batch_obs >= t_obs.shape[0]:
                    obs_sel = torch.arange(t_obs.shape[0], device=device)
                else:
                    obs_sel = torch.randperm(t_obs.shape[0], device=device)[:cfg.batch_obs]

                t_train = t_obs[obs_sel]
                y_train = y_data_all[ic_idx][obs_sel]
                xi_data = data_xi_bank[ic_idx][obs_sel]

                N_pred_train = sol_net(t_train)
                sigma_data = _sigma_for_data(
                    N_pred_train=N_pred_train,
                    y_train=y_train,
                    t_train=t_train,
                )
                data_resid = N_pred_train - y_train
                data_loss = randomized_residual_nll(
                    data_resid,
                    sigma=sigma_data,
                    xi=xi_data,
                )

                if torch.is_tensor(sigma_data):
                    total_sigma_data = total_sigma_data + sigma_data.detach().mean().item()
                else:
                    total_sigma_data = total_sigma_data + float(sigma_data)

                if cfg.batch_col >= cfg.n_col:
                    col_sel = torch.arange(cfg.n_col, device=device)
                else:
                    col_sel = torch.randperm(cfg.n_col, device=device)[:cfg.batch_col]

                t_col = t_col_bank[ic_idx][col_sel].clone().detach().requires_grad_(True)
                N_pred_col, _, _, phys_res = pinn_physics_residual(sol_net, dyn_net, t_col)
                xi_phys = xi_phys_bank[ic_idx][col_sel]

                sigma_phys = 0.1#_sigma_for_physics(N_pred_col, t_col)
                phys_loss = randomized_residual_nll(
                    phys_res,
                    sigma=sigma_phys,
                    xi=xi_phys,
                )

                if torch.is_tensor(sigma_phys):
                    total_sigma_phys = total_sigma_phys + sigma_phys.detach().mean().item()
                else:
                    total_sigma_phys = total_sigma_phys + float(sigma_phys)

                t0 = torch.zeros((1, 1), dtype=t_obs.dtype, device=device)
                y0_target = y_data_all[ic_idx][0:1]
                N0_pred = sol_net(t0)
                ic_loss = torch.mean((N0_pred - y0_target) ** 2)

                total_data_loss += data_loss
                total_phys_loss += phys_loss
                total_ic_loss += ic_loss

            total_data_loss /= n_ics
            total_phys_loss /= n_ics
            total_ic_loss /= n_ics
            mean_sigma_data = total_sigma_data / n_ics
            mean_sigma_phys = total_sigma_phys / n_ics

            dyn_reg = lambda_dyn_reg * _nn_weight_penalty(
                dyn_net,
                p=nn_reg_p,
                include_bias=nn_reg_include_bias,
                normalize=nn_reg_normalize,
            )
            sol_reg = lambda_sol_reg * _nn_weight_penalty(
                sol_nets,
                p=nn_reg_p,
                include_bias=nn_reg_include_bias,
                normalize=nn_reg_normalize,
            )
            reg_loss = dyn_reg + sol_reg

            pinn_loss = total_data_loss + cfg.lambda_phys * total_phys_loss
            optim_loss = pinn_loss + cfg.lambda_ic * total_ic_loss + reg_loss
            optim_loss.backward()
            optimizer.step()

            train_pinn_total_history.append(pinn_loss.item())
            train_data_history.append(total_data_loss.item())
            train_phys_history.append(total_phys_loss.item())
            train_ic_history.append(total_ic_loss.item())
            train_optim_total_history.append(optim_loss.item())
            train_reg_history.append(reg_loss.item())

            criterion = optim_loss.item()
            if criterion < best_train_loss:
                best_train_loss = criterion
                best_epoch = epoch
                best_state = {
                    "dyn_net": copy.deepcopy(dyn_net.state_dict()),
                    "sol_nets": [copy.deepcopy(sol_net.state_dict()) for sol_net in sol_nets],
                    "dyn_init_seed": dyn_seed,
                    "sol_init_seeds": sol_init_seeds.copy(),
                }

            tag = "MAP" if not randomized else "rPINN"
            if (epoch == 0) or ((epoch + 1) % cfg.print_every_pinn == 0) or (epoch == cfg.n_epochs_pinn - 1):
                sigma_msg = (
                    f"SigmaData~{mean_sigma_data:.4e} | SigmaPhys~{mean_sigma_phys:.4e}"
                    if use_learned_sigma
                    else f"SigmaData={sigma_data_const:.4e} | SigmaPhys={sigma_phys_const:.4e}"
                )
                print(
                    f"{tag} | Sample {sample_idx + 1 if randomized else 1}/"
                    f"{cfg.n_rpinn_samples if randomized else 1} | "
                    f"Epoch {epoch + 1}/{cfg.n_epochs_pinn} | "
                    f"PINN={pinn_loss.item():.6e} | Optim={optim_loss.item():.6e} | "
                    f"Data={total_data_loss.item():.6e} | Phys={total_phys_loss.item():.6e} | "
                    f"IC={total_ic_loss.item():.6e} | Reg={reg_loss.item():.6e} | {sigma_msg}"
                )

        dyn_net.load_state_dict(best_state["dyn_net"])
        for ic_idx in range(n_ics):
            sol_nets[ic_idx].load_state_dict(best_state["sol_nets"][ic_idx])

        dyn_net.eval()
        for sol_net in sol_nets:
            sol_net.eval()

        with torch.no_grad():
            trajs_this_run = []
            obs_fits_this_run = []

            for ic_idx in range(n_ics):
                trajs_this_run.append(sol_nets[ic_idx](t_plot).squeeze(1).cpu().numpy())
                obs_fits_this_run.append(sol_nets[ic_idx](t_obs).squeeze(1).cpu().numpy())

            H_grid_learned = dyn_net.h(N_grid).squeeze(1).cpu().numpy()
            G_grid_learned = dyn_net(N_grid).squeeze(1).cpu().numpy()

        return {
            "train_pinn_total_history": train_pinn_total_history,
            "train_data_history": train_data_history,
            "train_phys_history": train_phys_history,
            "train_ic_history": train_ic_history,
            "train_optim_total_history": train_optim_total_history,
            "train_reg_history": train_reg_history,
            "trajs": np.array(trajs_this_run),
            "obs_fits": np.array(obs_fits_this_run),
            "rhs": G_grid_learned,
            "h": H_grid_learned,
            "best_train_loss": best_train_loss,
            "best_epoch": best_epoch,
            "state": best_state,
        }

    map_result = _train_single_run(
        sample_idx=0,
        randomized=False,
        init_from_state=None,
        dyn_seed=getattr(cfg, "map_dyn_init_seed", cfg.dyn_init_seed_base),
        sol_seed_base=getattr(cfg, "map_sol_init_seed_base", cfg.sol_init_seed_base),
    )
    map_state = map_result["state"]

    for sample_idx in range(cfg.n_rpinn_samples):
        result = _train_single_run(
            sample_idx=sample_idx,
            randomized=True,
            init_from_state=map_state,
            dyn_seed=cfg.dyn_init_seed_base + sample_idx,
            sol_seed_base=cfg.sol_init_seed_base + 1000 * sample_idx,
        )

        train_pinn_total_histories.append(result["train_pinn_total_history"])
        train_data_histories.append(result["train_data_history"])
        train_phys_histories.append(result["train_phys_history"])
        train_ic_histories.append(result["train_ic_history"])
        train_optim_total_histories.append(result["train_optim_total_history"])
        train_reg_histories.append(result["train_reg_history"])

        traj_histories.append(result["trajs"])
        obs_fit_histories.append(result["obs_fits"])
        rhs_histories.append(result["rhs"])
        h_histories.append(result["h"])
        sample_states.append(
            {
                "best_train_loss": result["best_train_loss"],
                "best_epoch": result["best_epoch"],
                "state": result["state"],
                "uses_learned_sigma": use_learned_sigma,
                "sigma_eval_mode": sigma_eval_mode,
                "sigma_floor": sigma_floor,
                "sigma_data": None if use_learned_sigma else sigma_data_const,
                "sigma_phys": None if use_learned_sigma else sigma_phys_const,
            }
        )

    train_pinn_total_histories = np.array(train_pinn_total_histories)
    train_data_histories = np.array(train_data_histories)
    train_phys_histories = np.array(train_phys_histories)
    train_ic_histories = np.array(train_ic_histories)
    train_optim_total_histories = np.array(train_optim_total_histories)
    train_reg_histories = np.array(train_reg_histories)
    traj_histories = np.array(traj_histories)
    obs_fit_histories = np.array(obs_fit_histories)
    rhs_histories = np.array(rhs_histories)
    h_histories = np.array(h_histories)

    rhs_mean, rhs_lo, rhs_hi = mean_and_ci(rhs_histories)
    h_mean, h_lo, h_hi = mean_and_ci(h_histories)

    traj_mean_per_ic, traj_lo_per_ic, traj_hi_per_ic = [], [], []
    obs_fit_mean_per_ic, obs_fit_lo_per_ic, obs_fit_hi_per_ic = [], [], []

    for i in range(n_ics):
        m, lo, hi = mean_and_ci(traj_histories[:, i, :])
        traj_mean_per_ic.append(m)
        traj_lo_per_ic.append(lo)
        traj_hi_per_ic.append(hi)

        m, lo, hi = mean_and_ci(obs_fit_histories[:, i, :])
        obs_fit_mean_per_ic.append(m)
        obs_fit_lo_per_ic.append(lo)
        obs_fit_hi_per_ic.append(hi)

    return {
        "map_train_pinn_total_history": np.array(map_result["train_pinn_total_history"]),
        "map_train_data_history": np.array(map_result["train_data_history"]),
        "map_train_phys_history": np.array(map_result["train_phys_history"]),
        "map_train_ic_history": np.array(map_result["train_ic_history"]),
        "map_train_optim_total_history": np.array(map_result["train_optim_total_history"]),
        "map_train_reg_history": np.array(map_result["train_reg_history"]),
        "map_traj": map_result["trajs"],
        "map_obs_fit": map_result["obs_fits"],
        "map_rhs": map_result["rhs"],
        "map_h": map_result["h"],
        "map_best_train_loss": map_result["best_train_loss"],
        "map_best_epoch": map_result["best_epoch"],
        "map_state": map_state,
        "train_pinn_total_histories": train_pinn_total_histories,
        "train_data_histories": train_data_histories,
        "train_phys_histories": train_phys_histories,
        "train_ic_histories": train_ic_histories,
        "train_optim_total_histories": train_optim_total_histories,
        "train_reg_histories": train_reg_histories,
        "traj_histories": traj_histories,
        "obs_fit_histories": obs_fit_histories,
        "rhs_histories": rhs_histories,
        "rhs_mean": rhs_mean,
        "rhs_lo": rhs_lo,
        "rhs_hi": rhs_hi,
        "h_histories": h_histories,
        "h_mean": h_mean,
        "h_lo": h_lo,
        "h_hi": h_hi,
        "traj_mean_per_ic": traj_mean_per_ic,
        "traj_lo_per_ic": traj_lo_per_ic,
        "traj_hi_per_ic": traj_hi_per_ic,
        "obs_fit_mean_per_ic": obs_fit_mean_per_ic,
        "obs_fit_lo_per_ic": obs_fit_lo_per_ic,
        "obs_fit_hi_per_ic": obs_fit_hi_per_ic,
        "best_train_losses": np.array([x["best_train_loss"] for x in sample_states]),
        "best_epochs": np.array([x["best_epoch"] for x in sample_states]),
        "sample_states": sample_states,
        "uses_learned_sigma": use_learned_sigma,
        "sigma_eval_mode": sigma_eval_mode,
        "sigma_floor": sigma_floor,
        "sigma_data": None if use_learned_sigma else sigma_data_const,
        "sigma_phys": None if use_learned_sigma else sigma_phys_const,
    }


def fit_multi_ic_pinn(
    cfg,
    y_data_all,
    t_obs,
    t_plot,
    N_grid,
    n_ics,
    device,
    sigma_model=None,
    sigma_fit_result=None,
):
    return fit_multi_ic_rpinn(
        cfg=cfg,
        y_data_all=y_data_all,
        t_obs=t_obs,
        t_plot=t_plot,
        N_grid=N_grid,
        n_ics=n_ics,
        device=device,
        sigma_model=sigma_model,
        sigma_fit_result=sigma_fit_result,
    )