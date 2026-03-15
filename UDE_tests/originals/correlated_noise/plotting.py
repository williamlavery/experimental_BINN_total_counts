import matplotlib.pyplot as plt
import numpy as np


def plot_mean_abs_pct_error_vs_state(
    error_stats,
    show=True,
    save_prefix=None,
    key="mean_abs_pct_error_vs_state",
):
    """
    Plot mean absolute percentage error vs state N using binned statistics.
    """
    figs = {}

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(
        error_stats["bin_centers"],
        error_stats["mean_pct"],
        linewidth=2,
        label="mean abs % error",
    )

    ax.fill_between(
        error_stats["bin_centers"],
        error_stats["mean_pct"] - error_stats["std_pct"],
        error_stats["mean_pct"] + error_stats["std_pct"],
        alpha=0.25,
        label="±1 std",
    )

    ax.scatter(
        error_stats["N_flat"],
        error_stats["pct_flat"],
        s=12,
        alpha=0.15,
        label="individual points",
    )

    ax.set_xlabel("N")
    ax.set_ylabel("Mean absolute % error")
    ax.set_title("Percentage error vs state N")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    figs[key] = fig

    if save_prefix is not None:
        fig.savefig(f"{save_prefix}_{key}.png", dpi=200)

    if not show:
        plt.close(fig)

    return figs