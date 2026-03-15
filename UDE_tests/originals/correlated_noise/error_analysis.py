import numpy as np


def compute_mean_abs_pct_error_vs_state(y_true_all, y_data_all, n_bins=20):
    """
    Compute mean absolute percentage error binned by true state N.

    Parameters
    ----------
    y_true_all : torch.Tensor or np.ndarray
        Shape [n_ics, T, 1] or [n_ics, T]
    y_data_all : torch.Tensor or np.ndarray
        Shape [n_ics, T, 1] or [n_ics, T]
    n_bins : int
        Number of bins in N

    Returns
    -------
    dict with:
        y_true_np, y_data_np, noise_np, abs_noise_np, abs_pct_error_np,
        N_flat, pct_flat, bins, bin_centers, mean_pct, std_pct, counts
    """
    if hasattr(y_true_all, "detach"):
        y_true_np = y_true_all.squeeze(-1).detach().cpu().numpy()
    else:
        y_true_np = np.asarray(y_true_all).squeeze(-1)

    if hasattr(y_data_all, "detach"):
        y_data_np = y_data_all.squeeze(-1).detach().cpu().numpy()
    else:
        y_data_np = np.asarray(y_data_all).squeeze(-1)

    noise_np = y_data_np - y_true_np
    abs_noise_np = np.abs(noise_np)

    eps = 1e-12
    abs_pct_error_np = 100.0 * abs_noise_np / np.maximum(np.abs(y_true_np), eps)

    N_flat = y_true_np.reshape(-1)
    pct_flat = abs_pct_error_np.reshape(-1)

    bins = np.linspace(N_flat.min(), N_flat.max(), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    mean_pct = np.zeros(n_bins)
    std_pct = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (N_flat >= bins[i]) & (N_flat < bins[i + 1])
        else:
            mask = (N_flat >= bins[i]) & (N_flat <= bins[i + 1])

        if np.sum(mask) > 0:
            mean_pct[i] = pct_flat[mask].mean()
            std_pct[i] = pct_flat[mask].std(ddof=1) if np.sum(mask) > 1 else 0.0
            counts[i] = np.sum(mask)
        else:
            mean_pct[i] = np.nan
            std_pct[i] = np.nan
            counts[i] = 0

    return {
        "y_true_np": y_true_np,
        "y_data_np": y_data_np,
        "noise_np": noise_np,
        "abs_noise_np": abs_noise_np,
        "abs_pct_error_np": abs_pct_error_np,
        "N_flat": N_flat,
        "pct_flat": pct_flat,
        "bins": bins,
        "bin_centers": bin_centers,
        "mean_pct": mean_pct,
        "std_pct": std_pct,
        "counts": counts,
    }


def print_mean_abs_pct_error_by_bin(error_stats):
    print("\nMean percentage error by N bin")
    print("--------------------------------")
    for c, m, s, n in zip(
        error_stats["bin_centers"],
        error_stats["mean_pct"],
        error_stats["std_pct"],
        error_stats["counts"],
    ):
        if n > 0:
            print(
                f"N≈{c:.3f} | mean % error = {m:6.3f}% | "
                f"std = {s:6.3f}% | samples = {int(n)}"
            )