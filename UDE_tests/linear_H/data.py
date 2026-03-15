import numpy as np
import torch


def make_time_grids(cfg, device):
    t_obs = torch.linspace(cfg.t0, cfg.t1, cfg.numpts, device=device).view(-1, 1)
    t_plot = torch.linspace(cfg.t0, cfg.t1, cfg.n_plot, device=device).view(-1, 1)
    N_grid = torch.linspace(0.0, 1.1, cfg.n_grid, device=device).view(-1, 1)
    return t_obs, t_plot, N_grid


def H_true(N, r_true):
    return r_true * (1.0 - N)


def G_true(N, r_true):
    return N * H_true(N, r_true)


def rk4_step(f, y, t, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def odeint_rk4(f, y0, t):
    ys = [y0]
    y = y0
    for i in range(len(t) - 1):
        ti = t[i]
        dt = t[i + 1] - t[i]
        y = rk4_step(f, y, ti, dt)
        ys.append(y)
    return torch.stack(ys, dim=0)


def sample_ar1_noise_from_sigma(sigma_t, rho):
    sigma_t = sigma_t.reshape(-1, 1)
    T = sigma_t.shape[0]

    eps = torch.randn_like(sigma_t)
    noise = torch.zeros_like(sigma_t)

    noise[0] = sigma_t[0] * eps[0]
    scale = torch.sqrt(torch.tensor(max(1.0 - rho**2, 1e-8), device=sigma_t.device))

    for k in range(1, T):
        noise[k] = rho * noise[k - 1] + scale * sigma_t[k] * eps[k]

    return noise


def generate_synthetic_data(cfg, device):
    t_obs, t_plot, N_grid = make_time_grids(cfg, device)

    y_true_list = []
    y_data_list = []
    noise_list = []

    with torch.no_grad():
        for N0 in cfg.N0_list:
            y0_true = torch.tensor([[N0]], dtype=torch.float32, device=device)
            y_true = odeint_rk4(
                lambda t, y: y * H_true(y, cfg.r_true), y0_true, t_obs.squeeze()
            ).squeeze(1)

            sigma_t = cfg.noise_sig_coeff * y_true ** cfg.gamma
            noise_t = sample_ar1_noise_from_sigma(sigma_t, rho=cfg.rho_true)
            y_data = y_true + noise_t

            y_true_list.append(y_true)
            y_data_list.append(y_data)
            noise_list.append(noise_t)

        H_grid_true = H_true(N_grid, cfg.r_true)
        G_grid_true = N_grid * H_grid_true
        sigma_true_grid = cfg.noise_sig_coeff * N_grid ** cfg.gamma

    y_true_all = torch.stack(y_true_list, dim=0)
    y_data_all = torch.stack(y_data_list, dim=0)
    noise_all = torch.stack(noise_list, dim=0)

    return {
        "t_obs": t_obs,
        "t_plot": t_plot,
        "N_grid": N_grid,
        "y_true_all": y_true_all,
        "y_data_all": y_data_all,
        "noise_all": noise_all,
        "H_grid_true": H_grid_true,
        "G_grid_true": G_grid_true,
        "sigma_true_grid": sigma_true_grid,
    }


def compute_noise_summary(y_true_all, y_data_all, noise_sig_coeff, gamma):
    y_true_np = y_true_all.squeeze(-1).cpu().numpy()
    y_data_np = y_data_all.squeeze(-1).cpu().numpy()

    noise_np = y_data_np - y_true_np
    abs_noise_np = np.abs(noise_np)
    sigma_true_obs_np = (noise_sig_coeff * (y_true_all.squeeze(-1) ** gamma)).cpu().numpy()

    eps = 1e-12
    denom_true = np.maximum(np.abs(y_true_np), eps)
    denom_sym = np.maximum(np.abs(y_true_np) + np.abs(y_data_np), eps)

    pct_error_np = 100.0 * noise_np / denom_true
    abs_pct_error_np = 100.0 * abs_noise_np / denom_true
    smape_np = 100.0 * 2.0 * abs_noise_np / denom_sym
    z_np = noise_np / np.maximum(sigma_true_obs_np, eps)

    return {
        "y_true_np": y_true_np,
        "y_data_np": y_data_np,
        "noise_np": noise_np,
        "abs_noise_np": abs_noise_np,
        "sigma_true_obs_np": sigma_true_obs_np,
        "pct_error_np": pct_error_np,
        "abs_pct_error_np": abs_pct_error_np,
        "smape_np": smape_np,
        "z_np": z_np,
    }
