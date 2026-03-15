import copy
import numpy as np
import torch
import torch.nn as nn

from data import odeint_rk4
from losses import (
    ar1_gaussian_nll_batch,
    pinn_physics_residual,
    randomized_residual_nll,
    sigma_monotonicity_loss,
)
from models import DynamicsNet, SigmaNet, SolutionNet
from utils import mean_and_ci


@torch.no_grad()
def rollout_with_learned_rhs(dyn_net, y0, t):
    y_roll = odeint_rk4(lambda tt, yy: dyn_net(yy), y0, t.squeeze()).squeeze(1)
    return y_roll


def fit_sigma_ar1_repeated(
    N_seq_all,
    resid_seq_all,
    N_eval_grid,
    N_col_base,
    device,
    n_repeats=5,
    val_fraction=0.2,
    n_epochs=3000,
    lr=5e-3,
    hidden_dim=8,
    lambda_mon=1.0,
    lambda_reg=1e-5,
    seed_offset_split=5000,
    seed_offset_model=7000,
    label="sigma-ar1",
    print_every=500,
):
    B = N_seq_all.shape[0]
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

    val_total_histories = []
    val_nll_histories = []
    val_mon_histories = []

    curve_histories = []
    rho_histories = []

    best_val_losses = []
    best_epochs = []
    best_rhos = []
    best_models = []

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
        r_train = resid_seq_all[train_idx]

        if use_validation:
            N_val = N_seq_all[val_idx]
            r_val = resid_seq_all[val_idx]

        torch.manual_seed(seed_offset_model + repeat)
        np.random.seed(seed_offset_model + repeat)

        sigma_net = SigmaNet(hidden_dim=hidden_dim).to(device)
        rho_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))

        opt = torch.optim.Adam(list(sigma_net.parameters()) + [rho_raw], lr=lr)

        train_total_history = []
        train_nll_history = []
        train_mon_history = []

        val_total_history = []
        val_nll_history = []
        val_mon_history = []

        best_val_loss = float("inf")
        best_epoch = -1
        best_state = None

        for epoch in range(n_epochs):
            opt.zero_grad()

            rho = torch.tanh(rho_raw)
            sigma_pred_train = sigma_net(N_train)

            train_nll = ar1_gaussian_nll_batch(
                resid_all=r_train,
                sigma_all=sigma_pred_train,
                rho=rho,
                reduce="mean",
            )
            train_mon = sigma_monotonicity_loss(sigma_net, N_col_base)
            reg = lambda_reg * sum((p ** 2).sum() for p in sigma_net.parameters())

            train_total = train_nll + lambda_mon * train_mon + reg
            train_total.backward()
            opt.step()

            rho_eval = torch.tanh(rho_raw).detach()
            if use_validation:
                with torch.no_grad():
                    sigma_pred_val = sigma_net(N_val)
                    val_nll = ar1_gaussian_nll_batch(
                        resid_all=r_val,
                        sigma_all=sigma_pred_val,
                        rho=rho_eval,
                        reduce="mean",
                    )
                val_mon = sigma_monotonicity_loss(sigma_net, N_col_base)
                val_total = val_nll + lambda_mon * val_mon
                criterion = val_total.item()
            else:
                val_nll = torch.tensor(np.nan, device=device)
                val_mon = torch.tensor(np.nan, device=device)
                val_total = torch.tensor(np.nan, device=device)
                criterion = train_total.item()

            train_total_history.append(train_total.item())
            train_nll_history.append(train_nll.item())
            train_mon_history.append(train_mon.item())

            val_total_history.append(val_total.item())
            val_nll_history.append(val_nll.item())
            val_mon_history.append(val_mon.item())

            if (epoch == 0) or ((epoch + 1) % print_every == 0) or (epoch == n_epochs - 1):
                print(
                    f"{label} | Repeat {repeat + 1}/{n_repeats} | "
                    f"Epoch {epoch + 1}/{n_epochs} | rho={rho_eval.item():.4f} | "
                    f"Train Total={train_total.item():.6e} | Train NLL={train_nll.item():.6e} | "
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
        rho_best = torch.tanh(best_state["rho_raw"]).item()
        sigma_net.eval()

        with torch.no_grad():
            sigma_curve = sigma_net(N_eval_grid).squeeze(1).cpu().numpy()

        best_models.append(copy.deepcopy(sigma_net))
        train_total_histories.append(train_total_history)
        train_nll_histories.append(train_nll_history)
        train_mon_histories.append(train_mon_history)

        val_total_histories.append(val_total_history)
        val_nll_histories.append(val_nll_history)
        val_mon_histories.append(val_mon_history)

        curve_histories.append(sigma_curve)
        rho_histories.append(rho_best)

        best_val_losses.append(best_val_loss)
        best_epochs.append(best_epoch)
        best_rhos.append(rho_best)

    return {
        "train_total_histories": np.array(train_total_histories),
        "train_nll_histories": np.array(train_nll_histories),
        "train_mon_histories": np.array(train_mon_histories),
        "val_total_histories": np.array(val_total_histories),
        "val_nll_histories": np.array(val_nll_histories),
        "val_mon_histories": np.array(val_mon_histories),
        "curve_histories": np.array(curve_histories),
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

import copy
import numpy as np
import torch
import torch.nn as nn


def fit_multi_ic_rpinn(cfg, y_data_all, t_obs, t_plot, N_grid, n_ics, device):
    train_pinn_total_histories = []
    train_data_histories = []
    train_phys_histories = []
    train_ic_histories = []
    train_optim_total_histories = []

    traj_histories = []
    rhs_histories = []
    h_histories = []
    obs_fit_histories = []
    sample_states = []

    sigma_data = float(cfg.sigma_data_rpinn)
    sigma_phys = float(cfg.sigma_phys_rpinn)

    # -------------------------
    # Shared training routine
    # -------------------------
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
        dyn_net = DynamicsNet(hidden_dim=cfg.hidden_dim_dyn, factor_rhs_by_state=cfg.factor_rhs_by_state).to(device)

        sol_nets = []
        sol_init_seeds = []
        for ic_idx in range(n_ics):
            init_seed = sol_seed_base + ic_idx
            sol_init_seeds.append(init_seed)
            torch.manual_seed(init_seed)
            np.random.seed(init_seed)
            sol_nets.append(SolutionNet(hidden_dim=cfg.hidden_dim_sol).to(device))
        sol_nets = nn.ModuleList(sol_nets)

        # Optional warm start from MAP
        if init_from_state is not None:
            dyn_net.load_state_dict(copy.deepcopy(init_from_state["dyn_net"]))
            for ic_idx in range(n_ics):
                sol_nets[ic_idx].load_state_dict(copy.deepcopy(init_from_state["sol_nets"][ic_idx]))

        # Only needed for randomized stage
        if randomized:
            data_xi_bank, t_col_bank, xi_phys_bank = _build_randomization_banks(
                cfg=cfg,
                t_obs=t_obs,
                device=device,
                n_ics=n_ics,
                sample_idx=sample_idx,
            )
        else:
            # Deterministic MAP: xi = 0
            t_col_bank = []
            zero_data_xi_bank = []
            zero_phys_xi_bank = []
            for _ in range(n_ics):
                t_col_ic = torch.linspace(
                    float(t_obs.min()), float(t_obs.max()), cfg.n_col, device=device, dtype=t_obs.dtype
                ).reshape(-1, 1)
                t_col_bank.append(t_col_ic)
                zero_data_xi_bank.append(torch.zeros_like(t_obs))
                zero_phys_xi_bank.append(torch.zeros((cfg.n_col, 1), device=device, dtype=t_obs.dtype))
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

        best_train_loss = float("inf")
        best_epoch = -1
        best_state = None

        for epoch in range(cfg.n_epochs_pinn):
            optimizer.zero_grad()

            total_data_loss = 0.0
            total_phys_loss = 0.0
            total_ic_loss = 0.0

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
                data_resid = N_pred_train - y_train
                data_loss = randomized_residual_nll(data_resid, sigma=sigma_data, xi=xi_data)

                if cfg.batch_col >= cfg.n_col:
                    col_sel = torch.arange(cfg.n_col, device=device)
                else:
                    col_sel = torch.randperm(cfg.n_col, device=device)[:cfg.batch_col]

                t_col = t_col_bank[ic_idx][col_sel].clone().detach().requires_grad_(True)
                _, _, _, phys_res = pinn_physics_residual(sol_net, dyn_net, t_col)
                xi_phys = xi_phys_bank[ic_idx][col_sel]
                phys_loss = randomized_residual_nll(phys_res, sigma=sigma_phys, xi=xi_phys)


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

            pinn_loss = (
                total_data_loss
                + cfg.lambda_phys * total_phys_loss
            )
            optim_loss = pinn_loss + cfg.lambda_ic * total_ic_loss
            optim_loss.backward()
            optimizer.step()

            train_pinn_total_history.append(pinn_loss.item())
            train_data_history.append(total_data_loss.item())
            train_phys_history.append(total_phys_loss.item())
            train_ic_history.append(total_ic_loss.item())
            train_optim_total_history.append(optim_loss.item())

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
                print(
                    f"{tag} | Sample {sample_idx + 1 if randomized else 1}/"
                    f"{cfg.n_rpinn_samples if randomized else 1} | "
                    f"Epoch {epoch + 1}/{cfg.n_epochs_pinn} | "
                    f"PINN={pinn_loss.item():.6e} | Optim={optim_loss.item():.6e} | "
                    f"Data={total_data_loss.item():.6e} | Phys={total_phys_loss.item():.6e} | "
                    f"IC={total_ic_loss.item():.6e}"
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
            "trajs": np.array(trajs_this_run),
            "obs_fits": np.array(obs_fits_this_run),
            "rhs": G_grid_learned,
            "h": H_grid_learned,
            "best_train_loss": best_train_loss,
            "best_epoch": best_epoch,
            "state": best_state,
        }

    # -------------------------
    # 1) First compute MAP
    # -------------------------
    map_result = _train_single_run(
        sample_idx=0,
        randomized=False,
        init_from_state=None,
        dyn_seed=getattr(cfg, "map_dyn_init_seed", cfg.dyn_init_seed_base),
        sol_seed_base=getattr(cfg, "map_sol_init_seed_base", cfg.sol_init_seed_base),
    )
    map_state = map_result["state"]

    # -------------------------
    # 2) Then randomized samples, initialized from MAP
    # -------------------------
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

        traj_histories.append(result["trajs"])
        obs_fit_histories.append(result["obs_fits"])
        rhs_histories.append(result["rhs"])
        h_histories.append(result["h"])
        sample_states.append(
            {
                "best_train_loss": result["best_train_loss"],
                "best_epoch": result["best_epoch"],
                "state": result["state"],
                "sigma_data": sigma_data,
                "sigma_phys": sigma_phys,
            }
        )

    train_pinn_total_histories = np.array(train_pinn_total_histories)
    train_data_histories = np.array(train_data_histories)
    train_phys_histories = np.array(train_phys_histories)
    train_ic_histories = np.array(train_ic_histories)
    train_optim_total_histories = np.array(train_optim_total_histories)
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
        "sigma_data": sigma_data,
        "sigma_phys": sigma_phys,
    }


# Backward-compatible alias for callers still importing the old name.
def fit_multi_ic_pinn(cfg, y_data_all, t_obs, t_plot, N_grid, n_ics, device):
    return fit_multi_ic_rpinn(
        cfg=cfg,
        y_data_all=y_data_all,
        t_obs=t_obs,
        t_plot=t_plot,
        N_grid=N_grid,
        n_ics=n_ics,
        device=device,
    )