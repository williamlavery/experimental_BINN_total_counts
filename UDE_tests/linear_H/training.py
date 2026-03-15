import copy
import numpy as np
import torch
import torch.nn as nn

from data import odeint_rk4
from losses import (
    ar1_gaussian_nll_batch,
    h_zero_nonnegative_penalty,
    pinn_physics_residual,
    sigma_monotonicity_loss,
)
from models import DynamicsNet, SigmaNet, SolutionNet
from utils import mean_and_ci


@torch.no_grad()
def rollout_with_learned_rhs(dyn_net, y0, t):
    y_roll = odeint_rk4(lambda tt, yy: yy * dyn_net(yy), y0, t.squeeze()).squeeze(1)
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



def fit_multi_ic_pinn(cfg, y_data_all, t_obs, t_plot, N_grid, n_ics, device):
    train_pinn_total_histories = []
    train_data_histories = []
    train_phys_histories = []
    train_h0_histories = []
    train_ic_histories = []
    train_optim_total_histories = []

    val_total_histories = []
    val_eval_mask_histories = []

    traj_histories = []
    rhs_rollout_traj_histories = []          # NEW
    rhs_histories = []
    obs_fit_histories = []
    rhs_rollout_obs_fit_histories = []       # NEW

    best_val_losses = []
    best_epochs = []
    best_states = []

    lambda_h0 = getattr(cfg, "lambda_h0", 1.0)

    for split in range(n_ics):
        heldout_ic = split
        train_ic_idx = [i for i in range(n_ics) if i != heldout_ic]
        n_train_ics = len(train_ic_idx)

        torch.manual_seed(cfg.dyn_init_seed_base + split)
        np.random.seed(cfg.dyn_init_seed_base + split)
        dyn_net = DynamicsNet(hidden_dim=cfg.hidden_dim_dyn).to(device)

        sol_net_list = []
        sol_init_seeds_this_split = []
        for ic_idx in train_ic_idx:
            init_seed = cfg.sol_init_seed_base + 1000 * split + ic_idx
            sol_init_seeds_this_split.append(init_seed)
            torch.manual_seed(init_seed)
            np.random.seed(init_seed)
            sol_net_list.append(SolutionNet(hidden_dim=cfg.hidden_dim_sol).to(device))

        sol_nets = nn.ModuleList(sol_net_list)

        optimizer = torch.optim.Adam(
            list(dyn_net.parameters()) + list(sol_nets.parameters()),
            lr=cfg.learning_rate_pinn,
        )

        train_pinn_total_history = []
        train_data_history = []
        train_phys_history = []
        train_h0_history = []
        train_ic_history = []
        train_optim_total_history = []

        val_total_history = np.full(cfg.n_epochs_pinn, np.nan, dtype=np.float64)
        val_eval_mask = np.zeros(cfg.n_epochs_pinn, dtype=bool)

        best_val_loss = float("inf")
        best_epoch = -1
        best_state = None

        for epoch in range(cfg.n_epochs_pinn):
            optimizer.zero_grad()

            total_data_loss = 0.0
            total_phys_loss = 0.0
            total_h0_loss = 0.0
            total_ic_loss = 0.0

            for local_j, ic_idx in enumerate(train_ic_idx):
                sol_net = sol_nets[local_j]

                if cfg.batch_obs >= t_obs.shape[0]:
                    obs_sel = torch.arange(t_obs.shape[0], device=device)
                else:
                    obs_sel = torch.randperm(t_obs.shape[0], device=device)[:cfg.batch_obs]

                t_train = t_obs[obs_sel]
                y_train = y_data_all[ic_idx][obs_sel]

                N_pred_train = sol_net(t_train)
                data_loss = torch.mean((N_pred_train - y_train) ** 2)

                t_col = 6.0 * torch.rand(cfg.batch_col, 1, device=device)
                t_col.requires_grad_(True)
                _, _, _, phys_res = pinn_physics_residual(sol_net, dyn_net, t_col)
                phys_loss = torch.mean(phys_res ** 2)

                h0_loss = h_zero_nonnegative_penalty(dyn_net, device)

                t0 = torch.zeros((1, 1), dtype=t_obs.dtype, device=device)
                y0_target = y_data_all[ic_idx][0:1]
                N0_pred = sol_net(t0)
                ic_loss = torch.mean((N0_pred - y0_target) ** 2)

                total_data_loss += data_loss
                total_phys_loss += phys_loss
                total_h0_loss += h0_loss
                total_ic_loss += ic_loss

            total_data_loss /= n_train_ics
            total_phys_loss /= n_train_ics
            total_h0_loss /= n_train_ics
            total_ic_loss /= n_train_ics

            pinn_loss = (
                total_data_loss
                + cfg.lambda_phys * total_phys_loss
                + lambda_h0 * total_h0_loss
            )
            optim_loss = pinn_loss + cfg.lambda_ic * total_ic_loss

            optim_loss.backward()
            optimizer.step()

            train_pinn_total_history.append(pinn_loss.item())
            train_data_history.append(total_data_loss.item())
            train_phys_history.append(total_phys_loss.item())
            train_h0_history.append(total_h0_loss.item())
            train_ic_history.append(total_ic_loss.item())
            train_optim_total_history.append(optim_loss.item())

            do_validation = (
                (epoch == 0)
                or ((epoch + 1) % cfg.val_every == 0)
                or (epoch == cfg.n_epochs_pinn - 1)
            )

            current_val_loss = np.nan

            if do_validation:
                dyn_net.eval()
                for sol_net in sol_nets:
                    sol_net.eval()

                with torch.no_grad():
                    y0_val = y_data_all[heldout_ic][0:1]
                    y_val_target = y_data_all[heldout_ic]
                    y_val_roll = rollout_with_learned_rhs(dyn_net, y0_val, t_obs)

                    current_val_loss = torch.mean((y_val_roll - y_val_target) ** 2).item()
                    val_total_history[epoch] = current_val_loss
                    val_eval_mask[epoch] = True

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_epoch = epoch
                    best_state = {
                        "dyn_net": copy.deepcopy(dyn_net.state_dict()),
                        "sol_nets": [copy.deepcopy(sol_net.state_dict()) for sol_net in sol_nets],
                        "dyn_init_seed": cfg.dyn_init_seed_base + split,
                        "sol_init_seeds": sol_init_seeds_this_split.copy(),
                        "heldout_ic": heldout_ic,
                        "train_ic_idx": train_ic_idx.copy(),
                    }

                dyn_net.train()
                for sol_net in sol_nets:
                    sol_net.train()

            if (epoch == 0) or ((epoch + 1) % cfg.print_every_pinn == 0) or (epoch == cfg.n_epochs_pinn - 1):
                print(
                    f"PINN | Split {split + 1}/{n_ics} | Epoch {epoch + 1}/{cfg.n_epochs_pinn} | "
                    f"PINN={pinn_loss.item():.6e} | Optim={optim_loss.item():.6e} | "
                    f"Data={total_data_loss.item():.6e} | Phys={total_phys_loss.item():.6e} | "
                    f"H0={total_h0_loss.item():.6e} | IC={total_ic_loss.item():.6e} | "
                    f"Heldout={current_val_loss:.6e}"
                )

        dyn_net.load_state_dict(best_state["dyn_net"])
        for local_j in range(len(train_ic_idx)):
            sol_nets[local_j].load_state_dict(best_state["sol_nets"][local_j])

        dyn_net.eval()
        for sol_net in sol_nets:
            sol_net.eval()

        with torch.no_grad():
            trajs_this_split = []
            obs_fits_this_split = []
            rhs_rollout_trajs_this_split = []      # NEW
            rhs_rollout_obs_this_split = []        # NEW

            for ic_idx in range(n_ics):
                y0_ic = y_data_all[ic_idx][0:1]

                # NEW: rollout for every IC from best saved dyn_net
                traj_roll_all = rollout_with_learned_rhs(dyn_net, y0_ic, t_plot)
                obs_roll_all = rollout_with_learned_rhs(dyn_net, y0_ic, t_obs)
                rhs_rollout_trajs_this_split.append(traj_roll_all.squeeze(1).cpu().numpy())
                rhs_rollout_obs_this_split.append(obs_roll_all.squeeze(1).cpu().numpy())

                # Existing behavior kept unchanged
                if ic_idx == heldout_ic:
                    trajs_this_split.append(traj_roll_all.squeeze(1).cpu().numpy())
                    obs_fits_this_split.append(obs_roll_all.squeeze(1).cpu().numpy())
                else:
                    local_j = train_ic_idx.index(ic_idx)
                    trajs_this_split.append(sol_nets[local_j](t_plot).squeeze(1).cpu().numpy())
                    obs_fits_this_split.append(sol_nets[local_j](t_obs).squeeze(1).cpu().numpy())

            H_grid_learned = dyn_net(N_grid).squeeze(1).cpu().numpy()

        train_pinn_total_histories.append(train_pinn_total_history)
        train_data_histories.append(train_data_history)
        train_phys_histories.append(train_phys_history)
        train_h0_histories.append(train_h0_history)
        train_ic_histories.append(train_ic_history)
        train_optim_total_histories.append(train_optim_total_history)

        val_total_histories.append(val_total_history)
        val_eval_mask_histories.append(val_eval_mask)

        traj_histories.append(np.array(trajs_this_split))
        obs_fit_histories.append(np.array(obs_fits_this_split))
        rhs_rollout_traj_histories.append(np.array(rhs_rollout_trajs_this_split))   # NEW
        rhs_rollout_obs_fit_histories.append(np.array(rhs_rollout_obs_this_split))   # NEW
        rhs_histories.append(H_grid_learned)

        best_val_losses.append(best_val_loss)
        best_epochs.append(best_epoch)
        best_states.append(best_state)

    train_pinn_total_histories = np.array(train_pinn_total_histories)
    train_data_histories = np.array(train_data_histories)
    train_phys_histories = np.array(train_phys_histories)
    train_h0_histories = np.array(train_h0_histories)
    train_ic_histories = np.array(train_ic_histories)
    train_optim_total_histories = np.array(train_optim_total_histories)
    val_total_histories = np.array(val_total_histories)
    traj_histories = np.array(traj_histories)
    obs_fit_histories = np.array(obs_fit_histories)
    rhs_rollout_traj_histories = np.array(rhs_rollout_traj_histories)         # NEW
    rhs_rollout_obs_fit_histories = np.array(rhs_rollout_obs_fit_histories)   # NEW
    rhs_histories = np.array(rhs_histories)

    h_mean, h_lo, h_hi = mean_and_ci(rhs_histories)

    traj_mean_per_ic, traj_lo_per_ic, traj_hi_per_ic = [], [], []
    obs_fit_mean_per_ic, obs_fit_lo_per_ic, obs_fit_hi_per_ic = [], [], []

    rhs_rollout_traj_mean_per_ic, rhs_rollout_traj_lo_per_ic, rhs_rollout_traj_hi_per_ic = [], [], []   # NEW
    rhs_rollout_obs_fit_mean_per_ic, rhs_rollout_obs_fit_lo_per_ic, rhs_rollout_obs_fit_hi_per_ic = [], [], []   # NEW

    for i in range(n_ics):
        m, lo, hi = mean_and_ci(traj_histories[:, i, :])
        traj_mean_per_ic.append(m)
        traj_lo_per_ic.append(lo)
        traj_hi_per_ic.append(hi)

        m, lo, hi = mean_and_ci(obs_fit_histories[:, i, :])
        obs_fit_mean_per_ic.append(m)
        obs_fit_lo_per_ic.append(lo)
        obs_fit_hi_per_ic.append(hi)

        # NEW
        m, lo, hi = mean_and_ci(rhs_rollout_traj_histories[:, i, :])
        rhs_rollout_traj_mean_per_ic.append(m)
        rhs_rollout_traj_lo_per_ic.append(lo)
        rhs_rollout_traj_hi_per_ic.append(hi)

        m, lo, hi = mean_and_ci(rhs_rollout_obs_fit_histories[:, i, :])
        rhs_rollout_obs_fit_mean_per_ic.append(m)
        rhs_rollout_obs_fit_lo_per_ic.append(lo)
        rhs_rollout_obs_fit_hi_per_ic.append(hi)

    return {
        "train_pinn_total_histories": train_pinn_total_histories,
        "train_data_histories": train_data_histories,
        "train_phys_histories": train_phys_histories,
        "train_h0_histories": train_h0_histories,
        "train_ic_histories": train_ic_histories,
        "train_optim_total_histories": train_optim_total_histories,
        "val_total_histories": val_total_histories,
        "val_eval_mask_histories": np.array(val_eval_mask_histories),
        "traj_histories": traj_histories,
        "obs_fit_histories": obs_fit_histories,
        "rhs_rollout_traj_histories": rhs_rollout_traj_histories,                   # NEW
        "rhs_rollout_obs_fit_histories": rhs_rollout_obs_fit_histories,             # NEW
        "rhs_histories": rhs_histories,
        "rhs_mean": h_mean,
        "rhs_lo": h_lo,
        "rhs_hi": h_hi,
        "traj_mean_per_ic": traj_mean_per_ic,
        "traj_lo_per_ic": traj_lo_per_ic,
        "traj_hi_per_ic": traj_hi_per_ic,
        "obs_fit_mean_per_ic": obs_fit_mean_per_ic,
        "obs_fit_lo_per_ic": obs_fit_lo_per_ic,
        "obs_fit_hi_per_ic": obs_fit_hi_per_ic,
        "rhs_rollout_traj_mean_per_ic": rhs_rollout_traj_mean_per_ic,               # NEW
        "rhs_rollout_traj_lo_per_ic": rhs_rollout_traj_lo_per_ic,                   # NEW
        "rhs_rollout_traj_hi_per_ic": rhs_rollout_traj_hi_per_ic,                   # NEW
        "rhs_rollout_obs_fit_mean_per_ic": rhs_rollout_obs_fit_mean_per_ic,         # NEW
        "rhs_rollout_obs_fit_lo_per_ic": rhs_rollout_obs_fit_lo_per_ic,             # NEW
        "rhs_rollout_obs_fit_hi_per_ic": rhs_rollout_obs_fit_hi_per_ic,             # NEW
        "best_val_losses": np.array(best_val_losses),
        "best_epochs": np.array(best_epochs),
        "best_states": best_states,
    }