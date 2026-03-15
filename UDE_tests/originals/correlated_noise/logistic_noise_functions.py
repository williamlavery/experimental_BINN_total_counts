
import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# Reproducibility / device helpers
# ============================================================
def set_seed(seed=0):
    """Set NumPy and PyTorch seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device(device_str="cpu"):
    """Return a torch.device."""
    return torch.device(device_str)


# ============================================================
# 1. Ground-truth system: logistic growth dN/dt = r N (1 - N)
# ============================================================
def G_true(N, r_true=1.0):
    """Logistic growth RHS."""
    return r_true * N * (1.0 - N)


def make_default_grids(device=None, t_end=6.0, numpts=None, n_plot=200, n_grid=200):
    """
    Create standard time/state grids used by the simulation.
    """
    if device is None:
        device = torch.device("cpu")
    if numpts is None:
        numpts = 4 * (24 // 4)

    t_obs = torch.linspace(0.0, t_end, numpts, device=device).view(-1, 1)
    t_plot = torch.linspace(0.0, t_end, n_plot, device=device).view(-1, 1)
    N_grid = torch.linspace(0.0, 1.1, n_grid, device=device).view(-1, 1)
    return t_obs, t_plot, N_grid


# ============================================================
# AR(1) heteroscedastic Gaussian NLL
# ============================================================
def ar1_gaussian_nll(resid, sigma, rho, eps=1e-6, reduce="mean"):
    """
    AR(1) heteroscedastic Gaussian negative log-likelihood.

    Model:
        e_0 ~ N(0, sigma_0^2)
        e_k | e_{k-1} ~ N(rho e_{k-1}, (1-rho^2) sigma_k^2)
    """
    resid = resid.reshape(-1)
    sigma = sigma.reshape(-1).clamp_min(eps)

    sigma2 = sigma ** 2
    one_minus_rho2 = max(1.0 - rho ** 2, eps)

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

    B = resid_all.shape[0]
    losses = []
    for b in range(B):
        losses.append(
            ar1_gaussian_nll(resid_all[b], sigma_all[b], rho, eps=eps, reduce="mean")
        )
    losses = torch.stack(losses)

    if reduce == "mean":
        return losses.mean()
    if reduce == "sum":
        return losses.sum()
    raise ValueError("reduce must be 'mean' or 'sum'")


# ============================================================
# ODE integration
# ============================================================
def rk4_step(f, y, t, dt):
    """One RK4 step."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def odeint_rk4(f, y0, t):
    """Integrate an ODE with RK4 over time grid t."""
    ys = [y0]
    y = y0
    for i in range(len(t) - 1):
        ti = t[i]
        dt = t[i + 1] - t[i]
        y = rk4_step(f, y, ti, dt)
        ys.append(y)
    return torch.stack(ys, dim=0)


# ============================================================
# Noise generation
# ============================================================
def sample_ar1_noise_from_sigma(sigma_t, rho):
    """
    Sample heteroscedastic AR(1) noise with marginal std sigma_t.

    e_0 ~ N(0, sigma_0^2)
    e_k = rho e_{k-1} + sqrt(1-rho^2) sigma_k xi_k
    """
    sigma_t = sigma_t.reshape(-1, 1)
    T = sigma_t.shape[0]

    eps = torch.randn_like(sigma_t)
    noise = torch.zeros_like(sigma_t)

    noise[0] = sigma_t[0] * eps[0]
    scale = torch.sqrt(torch.tensor(max(1.0 - rho**2, 1e-8), device=sigma_t.device))

    for k in range(1, T):
        noise[k] = rho * noise[k - 1] + scale * sigma_t[k] * eps[k]

    return noise


def simulate_logistic_observations(
    N0_list,
    t_obs,
    r_true=1.0,
    noise_sig_coeff=0.1,
    gamma=1.0,
    rho_true=0.5,
    device=None,
):
    """
    Simulate true logistic trajectories plus heteroscedastic AR(1) observation noise.

    Returns a dictionary with tensors:
        y_true_all, y_data_all, noise_all, sigma_true_obs_all
    """
    if device is None:
        device = torch.device("cpu")

    y_true_list = []
    y_data_list = []
    noise_list = []
    sigma_list = []

    with torch.no_grad():
        for N0 in N0_list:
            y0_true = torch.tensor([[N0]], dtype=torch.float32, device=device)
            y_true = odeint_rk4(
                lambda t, y: G_true(y, r_true=r_true),
                y0_true,
                t_obs.squeeze(),
            ).squeeze(1)

            sigma_t = noise_sig_coeff * y_true**gamma
            noise_t = sample_ar1_noise_from_sigma(sigma_t, rho=rho_true)
            y_data = y_true + noise_t

            y_true_list.append(y_true)
            y_data_list.append(y_data)
            noise_list.append(noise_t)
            sigma_list.append(sigma_t)

    y_true_all = torch.stack(y_true_list, dim=0)   # [n_ics, T, 1]
    y_data_all = torch.stack(y_data_list, dim=0)   # [n_ics, T, 1]
    noise_all = torch.stack(noise_list, dim=0)     # [n_ics, T, 1]
    sigma_true_obs_all = torch.stack(sigma_list, dim=0)

    return {
        "y_true_all": y_true_all,
        "y_data_all": y_data_all,
        "noise_all": noise_all,
        "sigma_true_obs_all": sigma_true_obs_all,
    }


# ============================================================
# Noise statistics
# ============================================================
def compute_noise_statistics(y_true_all, y_data_all, noise_sig_coeff=0.1, gamma=1.0):
    """
    Compute pooled and pointwise noise statistics.

    Returns a dictionary containing NumPy arrays and summary scalars.
    """
    with torch.no_grad():
        y_true_np = y_true_all.squeeze(-1).cpu().numpy()
        y_data_np = y_data_all.squeeze(-1).cpu().numpy()

        noise_np = y_data_np - y_true_np
        abs_noise_np = np.abs(noise_np)

        sigma_true_obs_np = (
            noise_sig_coeff * (y_true_all.squeeze(-1) ** gamma)
        ).cpu().numpy()

        eps = 1e-12
        denom_true = np.maximum(np.abs(y_true_np), eps)
        denom_sym = np.maximum(np.abs(y_true_np) + np.abs(y_data_np), eps)

        pct_error_np = 100.0 * noise_np / denom_true
        abs_pct_error_np = 100.0 * abs_noise_np / denom_true
        smape_np = 100.0 * 2.0 * abs_noise_np / denom_sym
        z_np = noise_np / np.maximum(sigma_true_obs_np, eps)

        summary = {
            "mean_signed_noise": noise_np.mean(),
            "std_signed_noise": noise_np.std(ddof=1),
            "mean_absolute_noise": abs_noise_np.mean(),
            "median_absolute_noise": np.median(abs_noise_np),
            "rmse_noise": np.sqrt(np.mean(noise_np**2)),
            "max_absolute_noise": abs_noise_np.max(),
            "mpe_percent": pct_error_np.mean(),
            "mape_percent": abs_pct_error_np.mean(),
            "median_abs_percent_error": np.median(abs_pct_error_np),
            "smape_percent": smape_np.mean(),
            "mean_true_sigma": sigma_true_obs_np.mean(),
            "mean_abs_noise_over_sigma": np.mean(
                abs_noise_np / np.maximum(sigma_true_obs_np, eps)
            ),
            "std_standardized_noise": z_np.std(ddof=1),
            "mean_standardized_noise": z_np.mean(),
        }

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
        "summary": summary,
    }
