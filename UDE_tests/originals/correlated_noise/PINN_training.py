import copy
import numpy as np
import torch

from .losses import g_zero_nonnegative_penalty


def time_derivative(model, t):
    """
    Compute du/dt for a scalar-output network u(t).
    """
    t_req = t.clone().detach().requires_grad_(True)
    u = model(t_req)
    du_dt = torch.autograd.grad(
        outputs=u,
        inputs=t_req,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    return u, du_dt


def pinn_residual(solution_net, dynamics_net, t_col):
    """
    Compute PINN residual:
        du/dt - G(u)
    """
    u_col, du_dt_col = time_derivative(solution_net, t_col)
    G_u = dynamics_net(u_col)
    return du_dt_col - G_u


def data_fit_loss(solution_net, t_obs, y_obs):
    pred = solution_net(t_obs)
    return torch.mean((pred - y_obs) ** 2)


def ic_loss(solution_net, t0, y0):
    pred0 = solution_net(t0)
    return torch.mean((pred0 - y0) ** 2)


def physics_loss(solution_net, dynamics_net, t_col):
    r = pinn_residual(solution_net, dynamics_net, t_col)
    return torch.mean(r ** 2)


def fit_single_trajectory_pinn(
    t_obs,
    y_obs,
    t_col,
    solution_net,
    dynamics_net,
    n_epochs=5000,
    lr=1e-3,
    lambda_data=1.0,
    lambda_ic=1.0,
    lambda_phys=1.0,
    lambda_g0=0.0,
    print_every=500,
):
    """
    Fit a PINN to one observed trajectory.

    Parameters
    ----------
    t_obs : torch.Tensor
        Observation times, shape [T, 1]
    y_obs : torch.Tensor
        Observed states, shape [T, 1]
    t_col : torch.Tensor
        Collocation points, shape [T_col, 1]
    solution_net : nn.Module
    dynamics_net : nn.Module
    """
    device = t_obs.device
    opt = torch.optim.Adam(
        list(solution_net.parameters()) + list(dynamics_net.parameters()),
        lr=lr,
    )

    t0 = t_obs[:1]
    y0 = y_obs[:1]

    total_history = []
    data_history = []
    ic_history = []
    phys_history = []
    g0_history = []

    best_loss = float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(n_epochs):
        opt.zero_grad()

        loss_data = data_fit_loss(solution_net, t_obs, y_obs)
        loss_ic = ic_loss(solution_net, t0, y0)
        loss_phys = physics_loss(solution_net, dynamics_net, t_col)

        if lambda_g0 > 0.0:
            loss_g0 = g_zero_nonnegative_penalty(dynamics_net, device=device)
        else:
            loss_g0 = torch.tensor(0.0, device=device)

        loss_total = (
            lambda_data * loss_data
            + lambda_ic * loss_ic
            + lambda_phys * loss_phys
            + lambda_g0 * loss_g0
        )

        loss_total.backward()
        opt.step()

        total_history.append(loss_total.item())
        data_history.append(loss_data.item())
        ic_history.append(loss_ic.item())
        phys_history.append(loss_phys.item())
        g0_history.append(loss_g0.item())

        if (epoch == 0) or ((epoch + 1) % print_every == 0) or (epoch == n_epochs - 1):
            print(
                f"PINN | Epoch {epoch + 1:5d}/{n_epochs} | "
                f"Total: {loss_total.item():.6e} | "
                f"Data: {loss_data.item():.6e} | "
                f"IC: {loss_ic.item():.6e} | "
                f"Phys: {loss_phys.item():.6e} | "
                f"G0: {loss_g0.item():.6e}"
            )

        if loss_total.item() < best_loss:
            best_loss = loss_total.item()
            best_epoch = epoch
            best_state = {
                "solution_net": copy.deepcopy(solution_net.state_dict()),
                "dynamics_net": copy.deepcopy(dynamics_net.state_dict()),
            }

    solution_net.load_state_dict(best_state["solution_net"])
    dynamics_net.load_state_dict(best_state["dynamics_net"])

    return {
        "solution_net": solution_net,
        "dynamics_net": dynamics_net,
        "total_history": np.array(total_history),
        "data_history": np.array(data_history),
        "ic_history": np.array(ic_history),
        "phys_history": np.array(phys_history),
        "g0_history": np.array(g0_history),
        "best_loss": best_loss,
        "best_epoch": best_epoch,
    }


def fit_pinn_repeated(
    t_obs,
    y_obs,
    t_col,
    solution_net_ctor,
    dynamics_net_ctor,
    n_repeats=5,
    n_epochs=5000,
    lr=1e-3,
    lambda_data=1.0,
    lambda_ic=1.0,
    lambda_phys=1.0,
    lambda_g0=0.0,
    seed_offset=1000,
    print_every=500,
):
    """
    Repeat PINN fitting multiple times with different random seeds.
    """
    histories_total = []
    histories_data = []
    histories_ic = []
    histories_phys = []
    histories_g0 = []

    best_losses = []
    best_epochs = []
    best_solution_nets = []
    best_dynamics_nets = []

    for repeat in range(n_repeats):
        print(f"\n{'-' * 90}")
        print(f"Starting PINN repeat {repeat + 1}/{n_repeats}")
        print(f"{'-' * 90}")

        torch.manual_seed(seed_offset + repeat)
        np.random.seed(seed_offset + repeat)

        solution_net = solution_net_ctor().to(t_obs.device)
        dynamics_net = dynamics_net_ctor().to(t_obs.device)

        out = fit_single_trajectory_pinn(
            t_obs=t_obs,
            y_obs=y_obs,
            t_col=t_col,
            solution_net=solution_net,
            dynamics_net=dynamics_net,
            n_epochs=n_epochs,
            lr=lr,
            lambda_data=lambda_data,
            lambda_ic=lambda_ic,
            lambda_phys=lambda_phys,
            lambda_g0=lambda_g0,
            print_every=print_every,
        )

        histories_total.append(out["total_history"])
        histories_data.append(out["data_history"])
        histories_ic.append(out["ic_history"])
        histories_phys.append(out["phys_history"])
        histories_g0.append(out["g0_history"])

        best_losses.append(out["best_loss"])
        best_epochs.append(out["best_epoch"])
        best_solution_nets.append(copy.deepcopy(out["solution_net"]))
        best_dynamics_nets.append(copy.deepcopy(out["dynamics_net"]))

        print(
            f"Finished PINN repeat {repeat + 1:2d}/{n_repeats} | "
            f"Best epoch = {out['best_epoch']:5d} | "
            f"Best loss = {out['best_loss']:.6e}"
        )

    return {
        "total_histories": np.array(histories_total),
        "data_histories": np.array(histories_data),
        "ic_histories": np.array(histories_ic),
        "phys_histories": np.array(histories_phys),
        "g0_histories": np.array(histories_g0),
        "best_losses": np.array(best_losses),
        "best_epochs": np.array(best_epochs),
        "best_solution_nets": best_solution_nets,
        "best_dynamics_nets": best_dynamics_nets,
    }




def run_pinn_pipeline(
    t_obs,
    y_obs,
    t_col,
    solution_hidden_dim=32,
    dynamics_hidden_dim=32,
    n_repeats=3,
    n_epochs=5000,
    lr=1e-3,
    lambda_data=1.0,
    lambda_ic=1.0,
    lambda_phys=1.0,
    lambda_g0=0.0,
    print_every=500,
):
    return fit_pinn_repeated(
        t_obs=t_obs,
        y_obs=y_obs,
        t_col=t_col,
        solution_net_ctor=lambda: SolutionNet(hidden_dim=solution_hidden_dim),
        dynamics_net_ctor=lambda: DynamicsNet(hidden_dim=dynamics_hidden_dim),
        n_repeats=n_repeats,
        n_epochs=n_epochs,
        lr=lr,
        lambda_data=lambda_data,
        lambda_ic=lambda_ic,
        lambda_phys=lambda_phys,
        lambda_g0=lambda_g0,
        print_every=print_every,
    )