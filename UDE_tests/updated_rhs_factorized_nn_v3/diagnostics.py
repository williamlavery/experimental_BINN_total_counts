import numpy as np
import matplotlib.pyplot as plt


def plot_percentage_error_vs_state(y_true_all, y_data_all):
    y_true_np = y_true_all.squeeze(-1).cpu().numpy()
    y_data_np = y_data_all.squeeze(-1).cpu().numpy()

    noise_np = y_data_np - y_true_np
    abs_noise_np = np.abs(noise_np)

    eps = 1e-12
    abs_pct_error_np = 100.0 * abs_noise_np / np.maximum(np.abs(y_true_np), eps)

    N_flat = y_true_np.reshape(-1)
    pct_flat = abs_pct_error_np.reshape(-1)

    n_bins = 20
    bins = np.linspace(N_flat.min(), N_flat.max(), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    mean_pct = np.zeros(n_bins)
    std_pct = np.zeros(n_bins)

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (N_flat >= bins[i]) & (N_flat < bins[i + 1])
        else:
            mask = (N_flat >= bins[i]) & (N_flat <= bins[i + 1])

        if np.sum(mask) > 0:
            mean_pct[i] = pct_flat[mask].mean()
            std_pct[i] = pct_flat[mask].std(ddof=1) if np.sum(mask) > 1 else 0.0
        else:
            mean_pct[i] = np.nan
            std_pct[i] = np.nan

    plt.figure(figsize=(6, 4.5))
    plt.plot(bin_centers, mean_pct, linewidth=2, label="mean abs % error")
    plt.fill_between(bin_centers, mean_pct - std_pct, mean_pct + std_pct, alpha=0.25, label="±1 std")
    plt.scatter(N_flat, pct_flat, s=12, alpha=0.15, label="individual points")
    plt.xlabel("N")
    plt.ylabel("Mean absolute % error")
    plt.title("Percentage error vs state N")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
