
import numpy as np
import torch
import matplotlib.pyplot as plt


from logistic_noise_functions import (
    set_seed, get_device,
    make_default_grids, 
    simulate_logistic_observations,
    compute_noise_statistics,
    G_true, ar1_gaussian_nll, ar1_gaussian_nll_batch)

def print_noise_summary(stats, rho_true=None):
    """Pretty-print pooled noise statistics."""
    summary = stats["summary"]

    print("\nNoise summary: pooled over all ICs and times")
    print("--------------------------------------------")
    if rho_true is not None:
        print(f"AR(1) rho_true                     : {rho_true: .4f}")
    print(f"Mean signed noise                  : {summary['mean_signed_noise']: .6e}")
    print(f"Std of signed noise                : {summary['std_signed_noise']: .6e}")
    print(f"Mean absolute noise                : {summary['mean_absolute_noise']: .6e}")
    print(f"Median absolute noise              : {summary['median_absolute_noise']: .6e}")
    print(f"RMSE noise                         : {summary['rmse_noise']: .6e}")
    print(f"Max absolute noise                 : {summary['max_absolute_noise']: .6e}")
    print(f"Mean percentage error (MPE)        : {summary['mpe_percent']: .6f}%")
    print(f"Mean abs percentage error (MAPE)   : {summary['mape_percent']: .6f}%")
    print(f"Median abs percentage error        : {summary['median_abs_percent_error']: .6f}%")
    print(f"Symmetric MAPE (sMAPE)             : {summary['smape_percent']: .6f}%")
    print(f"Mean true sigma(N)                 : {summary['mean_true_sigma']: .6e}")
    print(f"Mean |noise| / sigma_true(N)       : {summary['mean_abs_noise_over_sigma']: .6f}")
    print(f"Std of standardized noise z        : {summary['std_standardized_noise']: .6f}")
    print(f"Mean of standardized noise z       : {summary['mean_standardized_noise']: .6f}")


def print_observed_point_table(t_obs, stats, N0_list):
    """Pretty-print pointwise table of true/data/noise/%error."""
    print("\nObserved-point noise table")
    print("--------------------------")
    for i in range(len(N0_list)):
        print(f"\nTrajectory {i + 1} (IC = {N0_list[i]:.2f})")
        print("t        true        data        noise       abs%err     sigma_true    z=noise/sigma")
        for ti, yt, yd, en, ape, sigi, zi in zip(
            t_obs.squeeze(1).cpu().numpy(),
            stats["y_true_np"][i],
            stats["y_data_np"][i],
            stats["noise_np"][i],
            stats["abs_pct_error_np"][i],
            stats["sigma_true_obs_np"][i],
            stats["z_np"][i],
        ):
            print(
                f"{ti:6.3f}   {yt:9.5f}   {yd:9.5f}   {en:9.5f}   "
                f"{ape:9.3f}%   {sigi:11.5e}   {zi:9.4f}"
            )


# ============================================================
# Plotting helpers
# ============================================================
# ============================================================
# Plotting helpers
# ============================================================
def moving_average_1d(x, window=3, center=True):
    """
    Compute a 1D moving average using NaN-padded edges so the output length
    matches the input length.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if window is None or window <= 1:
        return x.copy()
    if window > x.size:
        raise ValueError("window must be <= len(x)")

    kernel = np.ones(window, dtype=float) / float(window)
    valid = np.convolve(x, kernel, mode="valid")

    out = np.full_like(x, np.nan, dtype=float)
    if center:
        left = (window - 1) // 2
        right = left + valid.size
        out[left:right] = valid
    else:
        out[window - 1:] = valid
    return out


def moving_average_rows(arr, window=3, center=True):
    """
    Apply moving_average_1d row-wise to a 2D array of shape [n_rows, n_cols].
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    return np.vstack([moving_average_1d(row, window=window, center=center) for row in arr])


def average_across_ics(arr):
    """
    Average a [n_ics, T] array across initial conditions.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    return np.nanmean(arr, axis=0)


def compute_mean_and_ci(arr, ci_z=1.96):
    """
    Compute mean and pointwise CI across rows of a 2D array [n_ics, T].

    Returns:
        mean, lower, upper
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")

    mean = np.nanmean(arr, axis=0)
    n = np.sum(~np.isnan(arr), axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)

    sem = np.full_like(mean, np.nan, dtype=float)
    valid = n > 1
    sem[valid] = std[valid] / np.sqrt(n[valid])

    lower = mean - ci_z * sem
    upper = mean + ci_z * sem
    return mean, lower, upper


def sort_each_ic_by_state(x2d, y2d):
    """
    For state-based plots, sort each IC's x-values and reorder y-values to match.

    Returns:
        x_sorted, y_sorted
    """
    x2d = np.asarray(x2d, dtype=float)
    y2d = np.asarray(y2d, dtype=float)
    if x2d.shape != y2d.shape:
        raise ValueError("x2d and y2d must have the same shape")

    x_sorted = np.empty_like(x2d)
    y_sorted = np.empty_like(y2d)
    for i in range(x2d.shape[0]):
        idx = np.argsort(x2d[i])
        x_sorted[i] = x2d[i, idx]
        y_sorted[i] = y2d[i, idx]
    return x_sorted, y_sorted


def _finalize_figure(fig, key, figs, save_prefix=None, show=True):
    fig.tight_layout()
    figs[key] = fig
    if save_prefix is not None:
        fig.savefig(f"{save_prefix}_{key}.png", dpi=200)
    if not show:
        plt.close(fig)


def _plot_series_with_optional_moving_average(
    x2d,
    y2d,
    labels,
    xlabel,
    ylabel,
    title,
    key,
    save_prefix=None,
    show=True,
    add_zero_line=False,
    moving_average_window=None,
    moving_average_center=True,
    average_only=False,
    ci_z=1.96,
):
    """
    Generic helper for line plotting.

    Behavior:
      - Raw series are shown on the first axis.
      - If moving_average_window > 1, moving averages are shown on a separate subplot.
      - If average_only=True, plot mean across ICs and add pointwise CI bands.
    """
    figs = {}

    x2d = np.asarray(x2d, dtype=float)
    y2d = np.asarray(y2d, dtype=float)

    use_ma_subplot = moving_average_window is not None and moving_average_window > 1

    if use_ma_subplot:
        fig, axes = plt.subplots(
            2, 1, figsize=(8, 8), sharex=True,
            gridspec_kw={"height_ratios": [1, 1]}
        )
        ax_raw, ax_ma = axes
    else:
        fig, ax_raw = plt.subplots(figsize=(8, 5))
        ax_ma = None

    # -------------------------
    # Top subplot: raw series
    # -------------------------
    if average_only:
        x_mean = np.nanmean(x2d, axis=0)
        y_mean, y_lo, y_hi = compute_mean_and_ci(y2d, ci_z=ci_z)

        ax_raw.plot(x_mean, y_mean, marker="o", label="Average across IC")
        ax_raw.fill_between(x_mean, y_lo, y_hi, alpha=0.25, label="95% CI")
    else:
        for i, label in enumerate(labels):
            ax_raw.plot(x2d[i], y2d[i], marker="o", label=label)

    if add_zero_line:
        ax_raw.axhline(0.0, linestyle="--", linewidth=1)

    ax_raw.set_ylabel(ylabel)
    ax_raw.set_title(title if not use_ma_subplot else f"{title} (raw)")
    ax_raw.legend()
    ax_raw.grid(True, alpha=0.3)

    # -------------------------
    # Bottom subplot: moving averages
    # -------------------------
    if use_ma_subplot:
        y2d_ma = moving_average_rows(
            y2d, window=moving_average_window, center=moving_average_center
        )

        if average_only:
            x_mean = np.nanmean(x2d, axis=0)
            y_ma_mean, y_ma_lo, y_ma_hi = compute_mean_and_ci(y2d_ma, ci_z=ci_z)

            ax_ma.plot(
                x_mean,
                y_ma_mean,
                linewidth=2.5,
                marker="o",
                label=f"Average across IC (moving avg, w={moving_average_window})",
            )
            ax_ma.fill_between(x_mean, y_ma_lo, y_ma_hi, alpha=0.25, label="95% CI")
        else:
            for i, label in enumerate(labels):
                ax_ma.plot(
                    x2d[i],
                    y2d_ma[i],
                    linewidth=2.0,
                    marker="o",
                    label=f"{label} moving avg",
                )

        if add_zero_line:
            ax_ma.axhline(0.0, linestyle="--", linewidth=1)

        ax_ma.set_xlabel(xlabel)
        ax_ma.set_ylabel(f"{ylabel} (moving avg)")
        ax_ma.set_title(f"{title} (moving average, w={moving_average_window})")
        ax_ma.legend()
        ax_ma.grid(True, alpha=0.3)
    else:
        ax_raw.set_xlabel(xlabel)

    _finalize_figure(fig, key, figs, save_prefix=save_prefix, show=show)
    return figs


# ============================================================
# Plotting
# ============================================================


def plot_noise_vs_time(
    t_vals,
    noise,
    abs_noise,
    pct_noise,
    abs_pct_noise,
    N0_values,
    save_prefix=None,
    show=True,
    moving_average_window=None,
    moving_average_center=True,
):
    """
    Create separate plots:
      1) signed noise vs t
      2) absolute noise vs t
      3) signed percentage noise vs t
      4) absolute percentage noise vs t

    If moving_average_window > 1, the moving averages are drawn on a
    separate subplot within each figure.

    Returns a dictionary of figures.
    """
    t_vals = np.asarray(t_vals).reshape(-1)
    x2d = np.tile(t_vals[None, :], (len(N0_values), 1))
    labels = [f"IC={N0:.2f}" for N0 in N0_values]

    figs = {}
    figs.update(
        _plot_series_with_optional_moving_average(
            x2d=x2d,
            y2d=noise,
            labels=labels,
            xlabel="t",
            ylabel="Noise",
            title="Signed noise vs time",
            key="noise_vs_t_signed",
            save_prefix=save_prefix,
            show=show,
            add_zero_line=True,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
        )
    )
    figs.update(
        _plot_series_with_optional_moving_average(
            x2d=x2d,
            y2d=abs_noise,
            labels=labels,
            xlabel="t",
            ylabel="|Noise|",
            title="Absolute noise vs time",
            key="noise_vs_t_absolute",
            save_prefix=save_prefix,
            show=show,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
        )
    )
    figs.update(
        _plot_series_with_optional_moving_average(
            x2d=x2d,
            y2d=pct_noise,
            labels=labels,
            xlabel="t",
            ylabel="Noise (%)",
            title="Signed percentage noise vs time",
            key="noise_pct_vs_t_signed",
            save_prefix=save_prefix,
            show=show,
            add_zero_line=True,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
        )
    )
    figs.update(
        _plot_series_with_optional_moving_average(
            x2d=x2d,
            y2d=abs_pct_noise,
            labels=labels,
            xlabel="t",
            ylabel="|Noise| (%)",
            title="Absolute percentage noise vs time",
            key="noise_pct_vs_t_absolute",
            save_prefix=save_prefix,
            show=show,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
        )
    )

    return figs


def plot_noise_vs_N(
    y_true,
    noise,
    abs_noise,
    pct_noise,
    abs_pct_noise,
    N0_values,
    save_prefix=None,
    show=True,
    moving_average_window=None,
    moving_average_center=True,
):
    """
    Create separate plots:
      1) signed noise vs N
      2) absolute noise vs N
      3) signed percentage noise vs N
      4) absolute percentage noise vs N

    If moving_average_window > 1, the moving averages are drawn on a
    separate subplot within each figure.

    Returns a dictionary of figures.
    """
    labels = [f"IC={N0:.2f}" for N0 in N0_values]
    figs = {}

    x_sorted, y_sorted = sort_each_ic_by_state(y_true, noise)
    figs.update(
        _plot_series_with_optional_moving_average(
            x2d=x_sorted,
            y2d=y_sorted,
            labels=labels,
            xlabel="N (true state)",
            ylabel="Noise",
            title="Signed noise vs N",
            key="noise_vs_N_signed",
            save_prefix=save_prefix,
            show=show,
            add_zero_line=True,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
        )
    )

    x_sorted, y_sorted = sort_each_ic_by_state(y_true, abs_noise)
    figs.update(
        _plot_series_with_optional_moving_average(
            x2d=x_sorted,
            y2d=y_sorted,
            labels=labels,
            xlabel="N (true state)",
            ylabel="|Noise|",
            title="Absolute noise vs N",
            key="noise_vs_N_absolute",
            save_prefix=save_prefix,
            show=show,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
        )
    )

    x_sorted, y_sorted = sort_each_ic_by_state(y_true, pct_noise)
    figs.update(
        _plot_series_with_optional_moving_average(
            x2d=x_sorted,
            y2d=y_sorted,
            labels=labels,
            xlabel="N (true state)",
            ylabel="Noise (%)",
            title="Signed percentage noise vs N",
            key="noise_pct_vs_N_signed",
            save_prefix=save_prefix,
            show=show,
            add_zero_line=True,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
        )
    )

    x_sorted, y_sorted = sort_each_ic_by_state(y_true, abs_pct_noise)
    figs.update(
        _plot_series_with_optional_moving_average(
            x2d=x_sorted,
            y2d=y_sorted,
            labels=labels,
            xlabel="N (true state)",
            ylabel="|Noise| (%)",
            title="Absolute percentage noise vs N",
            key="noise_pct_vs_N_absolute",
            save_prefix=save_prefix,
            show=show,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
        )
    )

    return figs


def plot_average_noise_across_ics_vs_time(
    t_vals,
    noise,
    abs_noise,
    pct_noise,
    abs_pct_noise,
    save_prefix=None,
    show=True,
    moving_average_window=None,
    moving_average_center=True,
):
    """
    Plot the average across ICs as a function of time for:
      1) signed noise
      2) absolute noise
      3) signed percentage noise
      4) absolute percentage noise

    Adds pointwise 95% CI bands to the average plots. If moving_average_window > 1,
    the moving-average average is drawn on a separate subplot with its own CI band.

    Returns a dictionary of figures.
    """
    t_vals = np.asarray(t_vals).reshape(-1)
    n_ics = np.asarray(noise).shape[0]
    x2d = np.tile(t_vals[None, :], (n_ics, 1))
    labels = ["Average across IC"]

    figs = {}
    figs.update(
        _plot_series_with_optional_moving_average(
            x2d=x2d,
            y2d=noise,
            labels=labels,
            xlabel="t",
            ylabel="Average noise across IC",
            title="Average signed noise across IC vs time",
            key="avg_ic_noise_vs_t_signed",
            save_prefix=save_prefix,
            show=show,
            add_zero_line=True,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
            average_only=True,
        )
    )
    figs.update(
        _plot_series_with_optional_moving_average(
            x2d=x2d,
            y2d=abs_noise,
            labels=labels,
            xlabel="t",
            ylabel="Average |Noise| across IC",
            title="Average absolute noise across IC vs time",
            key="avg_ic_noise_vs_t_absolute",
            save_prefix=save_prefix,
            show=show,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
            average_only=True,
        )
    )
    figs.update(
        _plot_series_with_optional_moving_average(
            x2d=x2d,
            y2d=pct_noise,
            labels=labels,
            xlabel="t",
            ylabel="Average noise (%) across IC",
            title="Average signed percentage noise across IC vs time",
            key="avg_ic_noise_pct_vs_t_signed",
            save_prefix=save_prefix,
            show=show,
            add_zero_line=True,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
            average_only=True,
        )
    )
    figs.update(
        _plot_series_with_optional_moving_average(
            x2d=x2d,
            y2d=abs_pct_noise,
            labels=labels,
            xlabel="t",
            ylabel="Average |Noise| (%) across IC",
            title="Average absolute percentage noise across IC vs time",
            key="avg_ic_noise_pct_vs_t_absolute",
            save_prefix=save_prefix,
            show=show,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
            average_only=True,
        )
    )
    return figs


def plot_average_noise_across_ics_vs_N(
    y_true,
    noise,
    abs_noise,
    pct_noise,
    abs_pct_noise,
    save_prefix=None,
    show=True,
    moving_average_window=None,
    moving_average_center=True,
):
    """
    Plot the average across ICs as a function of N for:
      1) signed noise
      2) absolute noise
      3) signed percentage noise
      4) absolute percentage noise

    Adds pointwise 95% CI bands to the average plots. If moving_average_window > 1,
    the moving-average average is drawn on a separate subplot with its own CI band.

    Returns a dictionary of figures.
    """
    labels = ["Average across IC"]
    figs = {}

    x_sorted, y_sorted = sort_each_ic_by_state(y_true, noise)
    figs.update(
        _plot_series_with_optional_moving_average(
            x2d=x_sorted,
            y2d=y_sorted,
            labels=labels,
            xlabel="Average N across IC",
            ylabel="Average noise across IC",
            title="Average signed noise across IC vs N",
            key="avg_ic_noise_vs_N_signed",
            save_prefix=save_prefix,
            show=show,
            add_zero_line=True,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
            average_only=True,
        )
    )

    x_sorted, y_sorted = sort_each_ic_by_state(y_true, abs_noise)
    figs.update(
        _plot_series_with_optional_moving_average(
            x2d=x_sorted,
            y2d=y_sorted,
            labels=labels,
            xlabel="Average N across IC",
            ylabel="Average |Noise| across IC",
            title="Average absolute noise across IC vs N",
            key="avg_ic_noise_vs_N_absolute",
            save_prefix=save_prefix,
            show=show,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
            average_only=True,
        )
    )

    x_sorted, y_sorted = sort_each_ic_by_state(y_true, pct_noise)
    figs.update(
        _plot_series_with_optional_moving_average(
            x2d=x_sorted,
            y2d=y_sorted,
            labels=labels,
            xlabel="Average N across IC",
            ylabel="Average noise (%) across IC",
            title="Average signed percentage noise across IC vs N",
            key="avg_ic_noise_pct_vs_N_signed",
            save_prefix=save_prefix,
            show=show,
            add_zero_line=True,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
            average_only=True,
        )
    )

    x_sorted, y_sorted = sort_each_ic_by_state(y_true, abs_pct_noise)
    figs.update(
        _plot_series_with_optional_moving_average(
            x2d=x_sorted,
            y2d=y_sorted,
            labels=labels,
            xlabel="Average N across IC",
            ylabel="Average |Noise| (%) across IC",
            title="Average absolute percentage noise across IC vs N",
            key="avg_ic_noise_pct_vs_N_absolute",
            save_prefix=save_prefix,
            show=show,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
            average_only=True,
        )
    )

    return figs


def plot_average_noise_across_ics(
    stats,
    t_obs,
    save_prefix=None,
    show=True,
    moving_average_window=None,
    moving_average_center=True,
):
    """
    Convenience wrapper that generates all 'average across IC' plots
    versus both t and N.
    """
    figs_t = plot_average_noise_across_ics_vs_time(
        t_vals=t_obs.cpu().numpy(),
        noise=stats["noise_np"],
        abs_noise=stats["abs_noise_np"],
        pct_noise=stats["pct_error_np"],
        abs_pct_noise=stats["abs_pct_error_np"],
        save_prefix=save_prefix,
        show=show,
        moving_average_window=moving_average_window,
        moving_average_center=moving_average_center,
    )

    figs_N = plot_average_noise_across_ics_vs_N(
        y_true=stats["y_true_np"],
        noise=stats["noise_np"],
        abs_noise=stats["abs_noise_np"],
        pct_noise=stats["pct_error_np"],
        abs_pct_noise=stats["abs_pct_error_np"],
        save_prefix=save_prefix,
        show=show,
        moving_average_window=moving_average_window,
        moving_average_center=moving_average_center,
    )

    return {**figs_t, **figs_N}


def plot_all_noise_views(
    stats,
    t_obs,
    N0_list,
    save_prefix=None,
    show=True,
    moving_average_window=None,
    moving_average_center=True,
    include_average_across_ics=True,
):
    """
    Convenience wrapper that generates:
      - all per-IC noise plots
      - optionally all average-across-IC plots
    """
    figs_t = plot_noise_vs_time(
        t_vals=t_obs.cpu().numpy(),
        noise=stats["noise_np"],
        abs_noise=stats["abs_noise_np"],
        pct_noise=stats["pct_error_np"],
        abs_pct_noise=stats["abs_pct_error_np"],
        N0_values=N0_list,
        save_prefix=save_prefix,
        show=show,
        moving_average_window=moving_average_window,
        moving_average_center=moving_average_center,
    )

    figs_N = plot_noise_vs_N(
        y_true=stats["y_true_np"],
        noise=stats["noise_np"],
        abs_noise=stats["abs_noise_np"],
        pct_noise=stats["pct_error_np"],
        abs_pct_noise=stats["abs_pct_error_np"],
        N0_values=N0_list,
        save_prefix=save_prefix,
        show=show,
        moving_average_window=moving_average_window,
        moving_average_center=moving_average_center,
    )

    figs = {**figs_t, **figs_N}

    if include_average_across_ics:
        figs_avg = plot_average_noise_across_ics(
            stats=stats,
            t_obs=t_obs,
            save_prefix=save_prefix,
            show=show,
            moving_average_window=moving_average_window,
            moving_average_center=moving_average_center,
        )
        figs.update(figs_avg)

    return figs



# ============================================================
# Notebook-friendly workflow helper
# ============================================================
def run_noise_pipeline(
    N0_list,
    r_true=1.0,
    noise_sig_coeff=0.1,
    gamma=1.0,
    rho_true=0.5,
    t_end=6.0,
    numpts=None,
    device=None,
    seed=None,
):
    """
    End-to-end helper for notebook use.

    Returns a dictionary with:
      - t_obs, t_plot, N_grid
      - simulation outputs
      - noise statistics
    """
    if device is None:
        device = torch.device("cpu")
    if seed is not None:
        set_seed(seed)

    t_obs, t_plot, N_grid = make_default_grids(
        device=device,
        t_end=t_end,
        numpts=numpts,
    )

    sim = simulate_logistic_observations(
        N0_list=N0_list,
        t_obs=t_obs,
        r_true=r_true,
        noise_sig_coeff=noise_sig_coeff,
        gamma=gamma,
        rho_true=rho_true,
        device=device,
    )

    stats = compute_noise_statistics(
        y_true_all=sim["y_true_all"],
        y_data_all=sim["y_data_all"],
        noise_sig_coeff=noise_sig_coeff,
        gamma=gamma,
    )

    return {
        "t_obs": t_obs,
        "t_plot": t_plot,
        "N_grid": N_grid,
        "r_true": r_true,
        "rho_true": rho_true,
        "noise_sig_coeff": noise_sig_coeff,
        "gamma": gamma,
        **sim,
        "stats": stats,
    }
