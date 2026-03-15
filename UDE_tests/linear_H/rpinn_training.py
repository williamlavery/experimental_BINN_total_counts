import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from data import odeint_rk4
from losses import h_zero_nonnegative_penalty, pinn_physics_residual
from models import DynamicsNet, SolutionNet
from rpinn_losses import (
    ar1_gaussian_nll_fixed_sigma,
    gaussian_nll_fixed_sigma,
    parameter_anchor_penalty,
    sample_ar1_noise_from_sigma_batch,
)
from rpinn_models import posterior_mean_and_ci


class SolutionNet(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(init_weights_xavier)

    def forward(self, t):
        return self.net(t)
    
@dataclass
class RPINNConfig:
    device: str = "cpu"
    n_samples: int = 20
    n_epochs: int = 3000
    learning_rate: float = 1e-3
    batch_col: int = 128
    lambda_phys: float = 1.0
    lambda_anchor: float = 1.0
    lambda_h0: float = 1.0
    lambda_ic: float = 0.0
    prior_std: float = 1.0
    hidden_dim_sol: int = 32
    hidden_dim_dyn: int = 32
    dyn_init_seed_base: int = 30000
    sol_init_seed_base: int = 40000
    anchor_seed_base: int = 50000
    perturb_seed_base: int = 60000
    print_every: int = 250
    use_full_time_sequence_for_data: bool = True
    detach_sigma_inputs: bool = True


@torch.no_grad()
def rollout_with_learned_rhs(dyn_net: nn.Module, y0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return odeint_rk4(lambda tt, yy: yy * dyn_net(yy), y0, t.squeeze()).squeeze(1)



def _sample_module_anchors(module: nn.Module, prior_std: float) -> List[torch.Tensor]:
    return [prior_std * torch.randn_like(p) for p in module.parameters()]



def _build_sigma_inputs(pred: torch.Tensor, ref: torch.Tensor, detach_sigma_inputs: bool) -> torch.Tensor:
    x = pred if pred is not None else ref
    if detach_sigma_inputs:
        x = x.detach()
    return x



def build_randomized_targets(
    y_data_all: torch.Tensor,
    sigma_data_model: nn.Module,
    rho_data: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Create one perturbed observed-data target per rPINN sample.

    The perturbation uses the learned heteroskedastic sigma_data(N) and the
    learned AR(1) correlation rho_data.
    """
    with torch.no_grad():
        sigma_obs = sigma_data_model(y_data_all.to(device))
        noise_obs = sample_ar1_noise_from_sigma_batch(sigma_obs, rho=rho_data)
        y_tilde = y_data_all.to(device) + noise_obs
    return {
        "sigma_obs": sigma_obs,
        "noise_obs": noise_obs,
        "y_tilde": y_tilde,
    }



def fit_single_rpinn_sample(
    cfg: RPINNConfig,
    y_data_all: torch.Tensor,
    t_obs: torch.Tensor,
    t_plot: torch.Tensor,
    N_grid: torch.Tensor,
    sigma_data_model: nn.Module,
    rho_data: float,
    sigma_phys_model: nn.Module,
    sample_id: int,
) -> Dict[str, object]:
    device = torch.device(cfg.device)
    y_data_all = y_data_all.to(device)
    t_obs = t_obs.to(device)
    t_plot = t_plot.to(device)
    N_grid = N_grid.to(device)

    n_ics = y_data_all.shape[0]

    torch.manual_seed(cfg.perturb_seed_base + sample_id)
    np.random.seed(cfg.perturb_seed_base + sample_id)
    randomized = build_randomized_targets(y_data_all, sigma_data_model, rho_data, device)
    y_tilde_all = randomized["y_tilde"]

    torch.manual_seed(cfg.dyn_init_seed_base + sample_id)
    np.random.seed(cfg.dyn_init_seed_base + sample_id)
    dyn_net = DynamicsNet(hidden_dim=cfg.hidden_dim_dyn).to(device)

    sol_nets = []
    for ic_idx in range(n_ics):
        torch.manual_seed(cfg.sol_init_seed_base + 1000 * sample_id + ic_idx)
        np.random.seed(cfg.sol_init_seed_base + 1000 * sample_id + ic_idx)
        sol_nets.append(SolutionNet(hidden_dim=cfg.hidden_dim_sol).to(device))
    sol_nets = nn.ModuleList(sol_nets)

    torch.manual_seed(cfg.anchor_seed_base + sample_id)
    np.random.seed(cfg.anchor_seed_base + sample_id)
    dyn_anchor = _sample_module_anchors(dyn_net, prior_std=cfg.prior_std)
    sol_anchors = [_sample_module_anchors(sol_net, prior_std=cfg.prior_std) for sol_net in sol_nets]

    optimizer = torch.optim.Adam(list(dyn_net.parameters()) + list(sol_nets.parameters()), lr=cfg.learning_rate)

    history = {
        "total": [],
        "data_nll": [],
        "physics_nll": [],
        "anchor": [],
        "h0": [],
        "ic": [],
    }

    for epoch in range(cfg.n_epochs):
        optimizer.zero_grad()

        pred_seq_all = []
        for sol_net in sol_nets:
            pred_seq_all.append(sol_net(t_obs))
        pred_seq_all = torch.stack(pred_seq_all, dim=0)

        sigma_data_eval = sigma_data_model(
            _build_sigma_inputs(pred_seq_all, y_tilde_all, cfg.detach_sigma_inputs)
        )
        data_resid = pred_seq_all - y_tilde_all
        data_nll = ar1_gaussian_nll_fixed_sigma(
            resid_all=data_resid,
            sigma_all=sigma_data_eval,
            rho=rho_data,
            reduce="mean",
        )

        total_phys_nll = 0.0
        total_ic_loss = 0.0
        for ic_idx, sol_net in enumerate(sol_nets):
            t_col = t_obs.new_empty((cfg.batch_col, 1)).uniform_(float(t_obs.min()), float(t_obs.max()))
            t_col.requires_grad_(True)
            N_pred_col, _, _, phys_res = pinn_physics_residual(sol_net, dyn_net, t_col)

            # sigma_phys = sigma_phys_model(
            #                 _build_sigma_inputs(N_pred_col, N_pred_col, cfg.detach_sigma_inputs)
            #             )
            sigma_phys = 0.1
            eps_phys = sigma_phys.detach() * torch.randn_like(phys_res)
            phys_nll = gaussian_nll_fixed_sigma(phys_res - eps_phys, sigma_phys.detach(), reduce="mean")
            total_phys_nll = total_phys_nll + phys_nll

            t0 = torch.zeros((1, 1), dtype=t_obs.dtype, device=device)
            y0_target = y_tilde_all[ic_idx, 0:1]
            N0_pred = sol_net(t0)
            total_ic_loss = total_ic_loss + torch.mean((N0_pred - y0_target) ** 2)

        total_phys_nll = total_phys_nll / n_ics
        total_ic_loss = total_ic_loss / n_ics
        h0_loss = h_zero_nonnegative_penalty(dyn_net, device)
        anchor_pen = parameter_anchor_penalty([dyn_net] + list(sol_nets), [dyn_anchor] + sol_anchors, prior_std=cfg.prior_std)

        total_loss = (
            data_nll
            + cfg.lambda_phys * total_phys_nll
            + cfg.lambda_anchor * anchor_pen
            + cfg.lambda_h0 * h0_loss
            + cfg.lambda_ic * total_ic_loss
        )
        total_loss.backward()
        optimizer.step()

        history["total"].append(float(total_loss.item()))
        history["data_nll"].append(float(data_nll.item()))
        history["physics_nll"].append(float(total_phys_nll.item()))
        history["anchor"].append(float(anchor_pen.item()))
        history["h0"].append(float(h0_loss.item()))
        history["ic"].append(float(total_ic_loss.item()))

        if (epoch == 0) or ((epoch + 1) % cfg.print_every == 0) or (epoch == cfg.n_epochs - 1):
            print(
                f"rPINN | sample {sample_id + 1}/{cfg.n_samples} | epoch {epoch + 1}/{cfg.n_epochs} | "
                f"total={total_loss.item():.6e} | data_nll={data_nll.item():.6e} | "
                f"phys_nll={total_phys_nll.item():.6e} | anchor={anchor_pen.item():.6e}"
            )

    dyn_net.eval()
    for sol_net in sol_nets:
        sol_net.eval()

    with torch.no_grad():
        obs_fit_all = torch.stack([sol_net(t_obs) for sol_net in sol_nets], dim=0)
        traj_all = torch.stack([sol_net(t_plot) for sol_net in sol_nets], dim=0)
        H_grid = dyn_net(N_grid).squeeze(-1)
        sigma_data_grid = sigma_data_model(N_grid).squeeze(-1)
        sigma_phys_grid = sigma_phys_model(N_grid).squeeze(-1)
        rollout_all = torch.stack(
            [rollout_with_learned_rhs(dyn_net, y_data_all[ic_idx, 0:1], t_plot) for ic_idx in range(n_ics)],
            dim=0,
        )

    return {
        "dyn_state_dict": copy.deepcopy(dyn_net.state_dict()),
        "sol_state_dicts": [copy.deepcopy(sol_net.state_dict()) for sol_net in sol_nets],
        "history": history,
        "randomized": {k: v.detach().cpu() for k, v in randomized.items()},
        "obs_fit_all": obs_fit_all.detach().cpu(),
        "traj_all": traj_all.detach().cpu(),
        "rollout_all": rollout_all.detach().cpu(),
        "H_grid": H_grid.detach().cpu(),
        "sigma_data_grid": sigma_data_grid.detach().cpu(),
        "sigma_phys_grid": sigma_phys_grid.detach().cpu(),
    }



def fit_rpinn_ensemble(
    cfg: RPINNConfig,
    y_data_all: torch.Tensor,
    t_obs: torch.Tensor,
    t_plot: torch.Tensor,
    N_grid: torch.Tensor,
    sigma_data_model: nn.Module,
    rho_data: float,
    sigma_phys_model: nn.Module,
) -> Dict[str, object]:
    samples = []
    for sample_id in range(cfg.n_samples):
        out = fit_single_rpinn_sample(
            cfg=cfg,
            y_data_all=y_data_all,
            t_obs=t_obs,
            t_plot=t_plot,
            N_grid=N_grid,
            sigma_data_model=sigma_data_model,
            rho_data=rho_data,
            sigma_phys_model=sigma_phys_model,
            sample_id=sample_id,
        )
        samples.append(out)

    traj_stack = torch.stack([s["traj_all"] for s in samples], dim=0)
    rollout_stack = torch.stack([s["rollout_all"] for s in samples], dim=0)
    obs_fit_stack = torch.stack([s["obs_fit_all"] for s in samples], dim=0)
    H_stack = torch.stack([s["H_grid"] for s in samples], dim=0)

    traj_mean, traj_lo, traj_hi = posterior_mean_and_ci(traj_stack)
    rollout_mean, rollout_lo, rollout_hi = posterior_mean_and_ci(rollout_stack)
    obs_fit_mean, obs_fit_lo, obs_fit_hi = posterior_mean_and_ci(obs_fit_stack)
    H_mean, H_lo, H_hi = posterior_mean_and_ci(H_stack)

    return {
        "samples": samples,
        "traj_stack": traj_stack,
        "rollout_stack": rollout_stack,
        "obs_fit_stack": obs_fit_stack,
        "H_stack": H_stack,
        "traj_mean": traj_mean,
        "traj_lo": traj_lo,
        "traj_hi": traj_hi,
        "rollout_mean": rollout_mean,
        "rollout_lo": rollout_lo,
        "rollout_hi": rollout_hi,
        "obs_fit_mean": obs_fit_mean,
        "obs_fit_lo": obs_fit_lo,
        "obs_fit_hi": obs_fit_hi,
        "H_mean": H_mean,
        "H_lo": H_lo,
        "H_hi": H_hi,
    }
