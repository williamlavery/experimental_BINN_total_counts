import matplotlib.pyplot as plt
import numpy as np
import torch



def plot_rpinn_dynamics_posterior(
    t_plot,
    y_data_all,
    y_true_all,
    posterior_results,
    ic_idx=0,
    outpath=None,
):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    t_plot_np = torch.as_tensor(t_plot).detach().cpu().numpy().reshape(-1)

    y_data_np = torch.as_tensor(y_data_all).detach().cpu().numpy().squeeze(-1)
    y_true_np = torch.as_tensor(y_true_all).detach().cpu().numpy().squeeze(-1)

    mean_np = posterior_results["traj_mean"][ic_idx].detach().cpu().numpy().reshape(-1)
    lo_np = posterior_results["traj_lo"][ic_idx].detach().cpu().numpy().reshape(-1)
    hi_np = posterior_results["traj_hi"][ic_idx].detach().cpu().numpy().reshape(-1)

    # observation times
    t_obs = np.linspace(t_plot_np.min(), t_plot_np.max(), y_data_np.shape[1])

    plt.figure(figsize=(6, 4.5))

    # posterior band
    plt.fill_between(t_plot_np, lo_np, hi_np, alpha=0.25, label="95% posterior CI")

    # posterior mean
    plt.plot(t_plot_np, mean_np, linewidth=2, label="posterior mean")

    # noise-free trajectory
    plt.plot(
        t_obs,
        y_true_np[ic_idx],
        linestyle="--",
        linewidth=2,
        label="noise-free trajectory",
    )

    # observed noisy data
    plt.scatter(t_obs, y_data_np[ic_idx], s=22, label="observed data")

    plt.xlabel("t")
    plt.ylabel("N(t)")
    plt.title(f"rPINN trajectory posterior, IC {ic_idx}")
    plt.legend()
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=180, bbox_inches="tight")

    plt.show()



def plot_rpinn_H_posterior(N_grid, posterior_results, H_true=None, outpath=None):
    N_grid_np = torch.as_tensor(N_grid).detach().cpu().numpy().reshape(-1)
    H_mean = posterior_results["H_mean"].detach().cpu().numpy().reshape(-1)
    H_lo = posterior_results["H_lo"].detach().cpu().numpy().reshape(-1)
    H_hi = posterior_results["H_hi"].detach().cpu().numpy().reshape(-1)

    plt.figure(figsize=(6, 4.5))
    plt.fill_between(N_grid_np, H_lo, H_hi, alpha=0.25, label="95% posterior CI")
    plt.plot(N_grid_np, H_mean, linewidth=2, label="posterior mean H(N)")
    if H_true is not None:
        H_true_np = torch.as_tensor(H_true).detach().cpu().numpy().reshape(-1)
        plt.plot(N_grid_np, H_true_np, linestyle="--", linewidth=2, label="true H(N)")
    plt.xlabel("N")
    plt.ylabel("H(N)")
    plt.title("rPINN posterior for dynamics")
    plt.legend()
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.show()



def plot_randomized_targets_example(t_obs, y_data_all, posterior_results, sample_id=0, ic_idx=0, outpath=None):
    t_obs_np = torch.as_tensor(t_obs).detach().cpu().numpy().reshape(-1)
    y_obs_np = torch.as_tensor(y_data_all[ic_idx]).detach().cpu().numpy().reshape(-1)
    y_tilde_np = posterior_results["samples"][sample_id]["randomized"]["y_tilde"][ic_idx].numpy().reshape(-1)

    plt.figure(figsize=(6, 4.5))
    plt.plot(t_obs_np, y_obs_np, marker="o", label="original data")
    plt.plot(t_obs_np, y_tilde_np, marker="s", label="randomized target")
    plt.xlabel("t")
    plt.ylabel("N")
    plt.title(f"Randomized observed target, sample {sample_id}, IC {ic_idx}")
    plt.legend()
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.show()
