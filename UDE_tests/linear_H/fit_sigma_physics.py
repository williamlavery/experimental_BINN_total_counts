import copy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from config import ExperimentConfig
from data import generate_synthetic_data
from losses import gaussian_sigma_nll, pinn_physics_residual, sigma_monotonicity_loss
from models import DynamicsNet, SigmaNet, SolutionNet
from training import fit_multi_ic_pinn
from utils import mean_and_ci, set_seed


@dataclass
class PhysicsSigmaConfig:
    n_col_phys: int = 4000
    n_repeats: int = 5
    val_fraction: float = 0.2
    n_epochs: int = 3000
    lr: float = 5e-3
    hidden_dim: int = 16
    lambda_mon: float = 0.0
    lambda_reg: float = 1e-5
    split_seed_base: int = 15000
    model_seed_base: int = 17000
    print_every: int = 250
    n_bins_plot: int = 20


class SigmaPhysicsNet(nn.Module):
    """Same architecture as SigmaNet, renamed for clarity in plots/results."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.model = SigmaNet(hidden_dim=hidden_dim)

    def forward(self, N: torch.Tensor) -> torch.Tensor:
        return self.model(N)


def restore_best_split_models(best_state, cfg, device):
    dyn_net = DynamicsNet(hidden_dim=cfg.hidden_dim_dyn).to(device)
    dyn_net.load_state_dict(best_state["dyn_net"])
    dyn_net.eval()

    sol_nets = []
    for state in best_state["sol_nets"]:
        sol_net = SolutionNet(hidden_dim=cfg.hidden_dim_sol).to(device)
        sol_net.load_state_dict(state)
        sol_net.eval()
        sol_nets.append(sol_net)

    return dyn_net, sol_nets, best_state["train_ic_idx"], best_state["heldout_ic"]


def collect_physics_residual_dataset(cfg, pinn_results, device, n_col_phys=4000):
    """
    Build an empirical physics-residual dataset using fresh collocation points.

    For each trained split, we evaluate the residual
        r_phys(t) = dN/dt - N * H(N)
    on newly sampled collocation points for the solution nets that were actually
    trained in that split. This gives a set of (N_pred, residual) pairs that can
    be used to fit sigma_physics(N).
    """
    N_all = []
    resid_all = []
    split_ids = []
    ic_ids = []

    for split_id, best_state in enumerate(pinn_results["best_states"]):
        dyn_net, sol_nets, train_ic_idx, _ = restore_best_split_models(best_state, cfg, device)

        for local_j, ic_idx in enumerate(train_ic_idx):
            sol_net = sol_nets[local_j]

            t_col = cfg.t0 + (cfg.t1 - cfg.t0) * torch.rand(n_col_phys, 1, device=device)
            t_col.requires_grad_(True)

            N_pred, _, _, phys_res = pinn_physics_residual(sol_net, dyn_net, t_col)

            N_all.append(N_pred.detach())
            resid_all.append(phys_res.detach())
            split_ids.append(torch.full((n_col_phys, 1), split_id, device=device, dtype=torch.long))
            ic_ids.append(torch.full((n_col_phys, 1), ic_idx, device=device, dtype=torch.long))

    return {
        "N": torch.cat(N_all, dim=0),
        "resid": torch.cat(resid_all, dim=0),
        "split_id": torch.cat(split_ids, dim=0),
        "ic_id": torch.cat(ic_ids, dim=0),
    }


def fit_sigma_physics_repeated(
    N_all,
    resid_all,
    N_eval_grid,
    N_col_base,
    device,
    n_repeats=5,
    val_fraction=0.2,
    n_epochs=3000,
    lr=5e-3,
    hidden_dim=16,
    lambda_mon=0.0,
    lambda_reg=1e-5,
    split_seed_base=15000,
    model_seed_base=17000,
    print_every=250,
):
    n_total = N_all.shape[0]
    use_validation = val_fraction > 0.0 and n_total > 1

    if use_validation:
        n_val = int(round(val_fraction * n_total))
        n_val = max(1, min(n_val, n_total - 1))
    else:
        n_val = 0
    n_train = n_total - n_val

    train_total_histories = []
    train_nll_histories = []
    train_mon_histories = []
    val_total_histories = []
    val_nll_histories = []
    val_mon_histories = []
    curve_histories = []
    best_models = []
    best_epochs = []
    best_val_losses = []

    for repeat in range(n_repeats):
        g = torch.Generator(device=device)
        g.manual_seed(split_seed_base + repeat)
        perm = torch.randperm(n_total, generator=g, device=device)

        train_idx = perm[:n_train]
        if use_validation:
            val_idx = perm[n_train:]
        else:
            val_idx = torch.empty(0, dtype=torch.long, device=device)

        N_train = N_all[train_idx]
        r_train = resid_all[train_idx]
        if use_validation:
            N_val = N_all[val_idx]
            r_val = resid_all[val_idx]

        torch.manual_seed(model_seed_base + repeat)
        np.random.seed(model_seed_base + repeat)
        sigma_net = SigmaPhysicsNet(hidden_dim=hidden_dim).to(device)
        opt = torch.optim.Adam(sigma_net.parameters(), lr=lr)

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
            sigma_train = sigma_net(N_train)
            train_nll = gaussian_sigma_nll(r_train, sigma_train, reduce="mean")
            train_mon = sigma_monotonicity_loss(sigma_net, N_col_base) if lambda_mon > 0.0 else torch.tensor(0.0, device=device)
            reg = lambda_reg * sum((p ** 2).sum() for p in sigma_net.parameters())
            train_total = train_nll + lambda_mon * train_mon + reg
            train_total.backward()
            opt.step()

            if use_validation:
                with torch.no_grad():
                    sigma_val = sigma_net(N_val)
                    val_nll = gaussian_sigma_nll(r_val, sigma_val, reduce="mean")
                val_mon = sigma_monotonicity_loss(sigma_net, N_col_base) if lambda_mon > 0.0 else torch.tensor(0.0, device=device)
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
                    f"sigma-physics | repeat {repeat + 1}/{n_repeats} | "
                    f"epoch {epoch + 1}/{n_epochs} | "
                    f"train_total={train_total.item():.6e} | "
                    f"train_nll={train_nll.item():.6e} | "
                    f"val_total={val_total.item():.6e}"
                )

            if criterion < best_val_loss:
                best_val_loss = criterion
                best_epoch = epoch
                best_state = copy.deepcopy(sigma_net.state_dict())

        sigma_net.load_state_dict(best_state)
        sigma_net.eval()
        with torch.no_grad():
            sigma_curve = sigma_net(N_eval_grid).squeeze(1).cpu().numpy()

        train_total_histories.append(train_total_history)
        train_nll_histories.append(train_nll_history)
        train_mon_histories.append(train_mon_history)
        val_total_histories.append(val_total_history)
        val_nll_histories.append(val_nll_history)
        val_mon_histories.append(val_mon_history)
        curve_histories.append(sigma_curve)
        best_models.append(copy.deepcopy(sigma_net))
        best_epochs.append(best_epoch)
        best_val_losses.append(best_val_loss)

    curve_histories = np.array(curve_histories)
    curve_mean, curve_lo, curve_hi = mean_and_ci(curve_histories)

    return {
        "train_total_histories": np.array(train_total_histories),
        "train_nll_histories": np.array(train_nll_histories),
        "train_mon_histories": np.array(train_mon_histories),
        "val_total_histories": np.array(val_total_histories),
        "val_nll_histories": np.array(val_nll_histories),
        "val_mon_histories": np.array(val_mon_histories),
        "curve_histories": curve_histories,
        "curve_mean": curve_mean,
        "curve_lo": curve_lo,
        "curve_hi": curve_hi,
        "best_models": best_models,
        "best_epochs": np.array(best_epochs),
        "best_val_losses": np.array(best_val_losses),
    }


def compute_binned_rms(N_all, resid_all, n_bins=20):
    N_np = N_all.reshape(-1)
    r_np = resid_all.reshape(-1)

    bins = np.linspace(N_np.min(), N_np.max(), n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    rms = np.full(n_bins, np.nan)
    std_abs = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (N_np >= bins[i]) & (N_np < bins[i + 1])
        else:
            mask = (N_np >= bins[i]) & (N_np <= bins[i + 1])

        counts[i] = int(mask.sum())
        if counts[i] > 0:
            rr = r_np[mask]
            rms[i] = np.sqrt(np.mean(rr ** 2))
            std_abs[i] = np.std(np.abs(rr), ddof=1) if counts[i] > 1 else 0.0

    return centers, rms, std_abs, counts


def plot_sigma_physics_fit(dataset, N_grid, sigma_results, outpath=None, n_bins=20):
    N_np = dataset["N"].detach().cpu().numpy().reshape(-1)
    resid_np = dataset["resid"].detach().cpu().numpy().reshape(-1)
    abs_resid_np = np.abs(resid_np)

    N_grid_np = N_grid.detach().cpu().numpy().reshape(-1)
    sigma_mean = sigma_results["curve_mean"]
    sigma_lo = sigma_results["curve_lo"]
    sigma_hi = sigma_results["curve_hi"]

    centers, rms, _, counts = compute_binned_rms(N_np, resid_np, n_bins=n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(N_np, abs_resid_np, s=6, alpha=0.12, label="|physics residual|")
    axes[0].plot(N_grid_np, sigma_mean, linewidth=2.5, label=r"$\hat\sigma_{phys}(N)$")
    axes[0].fill_between(N_grid_np, sigma_lo, sigma_hi, alpha=0.25, label="95% ensemble CI")
    axes[0].scatter(centers, rms, s=28, label="binned RMS residual")
    axes[0].set_xlabel("Predicted state N")
    axes[0].set_ylabel("Residual scale")
    axes[0].set_title("Physics uncertainty fit from collocation residuals")
    axes[0].legend(fontsize=8)

    axes[1].hist(resid_np, bins=60, density=True, alpha=0.7)
    axes[1].set_xlabel("Physics residual")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Distribution of collocation residuals")

    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.show()

    print("Binned collocation counts:")
    print(counts)


