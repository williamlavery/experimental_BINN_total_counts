import copy
import numpy as np
import torch
import torch.nn as nn

from losses import ar1_gaussian_nll_batch, sigma_monotonicity_loss
from models import SigmaNet


def fit_sigma_repeated(
    N_all_seq,              # [B, T, 1]
    resid_all_seq,          # [B, T, 1]
    N_eval_grid,
    N_col_base,
    device,
    rho_assumed=0.85,
    n_repeats=1,
    val_fraction=0.2,
    n_epochs=1000,
    lr=5e-3,
    hidden_dim=8,
    lambda_mon=1.0,
    lambda_reg=1e-5,
    seed_offset_split=5000,
    seed_offset_model=7000,
    label="sigma",
    print_every=500,
):
    B = N_all_seq.shape[0]
    use_validation = val_fraction > 0.0 and B > 1

    if use_validation:
        n_val = int(round(val_fraction * B))
        n_val = max(1, min(n_val, B - 1))
    else:
        n_val = 0
    n_train = B - n_val

    train_total_histories = []
    train_nll_histories = []
    train_mon_histories = []

    val_total_histories = []
    val_nll_histories = []
    val_mon_histories = []

    curve_histories = []
    best_val_losses = []
    best_epochs = []
    best_models = []

    for repeat in range(n_repeats):
        print(f"\n{'-' * 90}")
        print(f"Starting {label} repeat {repeat + 1}/{n_repeats}")
        print(f"{'-' * 90}")

        g = torch.Generator(device=device)
        g.manual_seed(seed_offset_split + repeat)

        perm = torch.randperm(B, generator=g, device=device)
        train_idx = perm[:n_train].sort().values

        if use_validation:
            val_idx = perm[n_train:].sort().values
        else:
            val_idx = torch.empty(0, dtype=torch.long, device=device)

        N_train = N_all_seq[train_idx]
        r_train = resid_all_seq[train_idx]

        if use_validation:
            N_val = N_all_seq[val_idx]
            r_val = resid_all_seq[val_idx]

        torch.manual_seed(seed_offset_model + repeat)
        np.random.seed(seed_offset_model + repeat)

        sigma_net = SigmaNet(hidden_dim=hidden_dim).to(device)
        opt = torch.optim.Adam(sigma_net.parameters(), lr=lr)

        train_total_history = []
        train_nll_history = []
        train_mon_history = []

        val_total_history = []
        val_nll_history = []
        val_mon_history = []

        best_criterion = float("inf")
        best_epoch = -1
        best_state = None

        for epoch in range(n_epochs):
            opt.zero_grad()

            sigma_pred_train = sigma_net(N_train)
            train_nll = ar1_gaussian_nll_batch(
                r_train, sigma_pred_train, rho=rho_assumed, reduce="mean"
            )
            train_mon = sigma_monotonicity_loss(sigma_net, N_col_base)
            reg = lambda_reg * sum((p ** 2).sum() for p in sigma_net.parameters())

            train_total = train_nll + lambda_mon * train_mon + reg
            train_total.backward()
            opt.step()

            if use_validation:
                with torch.no_grad():
                    sigma_pred_val = sigma_net(N_val)
                    val_nll = ar1_gaussian_nll_batch(
                        r_val, sigma_pred_val, rho=rho_assumed, reduce="mean"
                    )

                val_mon = sigma_monotonicity_loss(sigma_net, N_col_base)
                val_total = val_nll + lambda_mon * val_mon

                criterion_value = val_total.item()
                val_total_item = val_total.item()
                val_nll_item = val_nll.item()
                val_mon_item = val_mon.item()
            else:
                criterion_value = train_total.item()
                val_total_item = np.nan
                val_nll_item = np.nan
                val_mon_item = np.nan

            train_total_history.append(train_total.item())
            train_nll_history.append(train_nll.item())
            train_mon_history.append(train_mon.item())

            val_total_history.append(val_total_item)
            val_nll_history.append(val_nll_item)
            val_mon_history.append(val_mon_item)

            if (epoch == 0) or ((epoch + 1) % print_every == 0) or (epoch == n_epochs - 1):
                if use_validation:
                    print(
                        f"{label} | Repeat {repeat + 1:2d}/{n_repeats} | "
                        f"Epoch {epoch + 1:5d}/{n_epochs} | "
                        f"Train Total: {train_total.item():.6e} | "
                        f"Train NLL: {train_nll.item():.6e} | "
                        f"Train Mon: {train_mon.item():.6e} | "
                        f"Val Total: {val_total_item:.6e} | "
                        f"Val NLL: {val_nll_item:.6e} | "
                        f"Val Mon: {val_mon_item:.6e}"
                    )
                else:
                    print(
                        f"{label} | Repeat {repeat + 1:2d}/{n_repeats} | "
                        f"Epoch {epoch + 1:5d}/{n_epochs} | "
                        f"Train Total: {train_total.item():.6e} | "
                        f"Train NLL: {train_nll.item():.6e} | "
                        f"Train Mon: {train_mon.item():.6e}"
                    )

            if criterion_value < best_criterion:
                best_criterion = criterion_value
                best_epoch = epoch
                best_state = copy.deepcopy(sigma_net.state_dict())

        sigma_net.load_state_dict(best_state)
        sigma_net.eval()

        with torch.no_grad():
            sigma_curve = sigma_net(N_eval_grid).squeeze(1).cpu().numpy()

        best_models.append(copy.deepcopy(sigma_net))
        train_total_histories.append(train_total_history)
        train_nll_histories.append(train_nll_history)
        train_mon_histories.append(train_mon_history)

        val_total_histories.append(val_total_history)
        val_nll_histories.append(val_nll_history)
        val_mon_histories.append(val_mon_history)

        curve_histories.append(sigma_curve)
        best_val_losses.append(best_criterion)
        best_epochs.append(best_epoch)

        if use_validation:
            print(
                f"Finished {label} repeat {repeat + 1:2d}/{n_repeats} | "
                f"Best epoch = {best_epoch:5d} | Best val loss = {best_criterion:.6e}"
            )
        else:
            print(
                f"Finished {label} repeat {repeat + 1:2d}/{n_repeats} | "
                f"Best epoch = {best_epoch:5d} | Best train loss = {best_criterion:.6e}"
            )

    return {
        "train_total_histories": np.array(train_total_histories),
        "train_nll_histories": np.array(train_nll_histories),
        "train_mon_histories": np.array(train_mon_histories),
        "val_total_histories": np.array(val_total_histories),
        "val_nll_histories": np.array(val_nll_histories),
        "val_mon_histories": np.array(val_mon_histories),
        "curve_histories": np.array(curve_histories),
        "best_val_losses": np.array(best_val_losses),
        "best_epochs": np.array(best_epochs),
        "best_models": best_models,
    }


def fit_sigma_ar1_repeated(
    N_seq_all,                # [B, T, 1]
    resid_seq_all,            # [B, T, 1]
    N_eval_grid,
    N_col_base,
    device,
    n_repeats=5,
    val_fraction=0.2,
    n_epochs=3000,
    lr=5e-3,
    hidden_dim=8,
    lambda_mon=1.0,
    lambda_reg=1e-5,
    seed_offset_split=5000,
    seed_offset_model=7000,
    label="sigma-ar1",
    print_every=500,
):
    B = N_seq_all.shape[0]
    use_validation = val_fraction > 0.0 and B > 1

    if use_validation:
        n_val = int(round(val_fraction * B))
        n_val = max(1, min(n_val, B - 1))
    else:
        n_val = 0
    n_train = B - n_val

    train_total_histories = []
    train_nll_histories = []
    train_mon_histories = []

    val_total_histories = []
    val_nll_histories = []
    val_mon_histories = []

    curve_histories = []
    rho_histories = []

    best_val_losses = []
    best_epochs = []
    best_rhos = []
    best_models = []

    for repeat in range(n_repeats):
        print(f"\n{'-' * 90}")
        print(f"Starting {label} repeat {repeat + 1}/{n_repeats}")
        print(f"{'-' * 90}")

        g = torch.Generator(device=device)
        g.manual_seed(seed_offset_split + repeat)

        perm = torch.randperm(B, generator=g, device=device)
        train_idx = perm[:n_train].sort().values

        if use_validation:
            val_idx = perm[n_train:].sort().values
        else:
            val_idx = torch.empty(0, dtype=torch.long, device=device)

        N_train = N_seq_all[train_idx]
        r_train = resid_seq_all[train_idx]

        if use_validation:
            N_val = N_seq_all[val_idx]
            r_val = resid_seq_all[val_idx]

        torch.manual_seed(seed_offset_model + repeat)
        np.random.seed(seed_offset_model + repeat)

        sigma_net = SigmaNet(hidden_dim=hidden_dim).to(device)
        rho_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))

        opt = torch.optim.Adam(
            list(sigma_net.parameters()) + [rho_raw],
            lr=lr,
        )

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

            rho = torch.tanh(rho_raw)
            sigma_pred_train = sigma_net(N_train)

            train_nll = ar1_gaussian_nll_batch(
                resid_all=r_train,
                sigma_all=sigma_pred_train,
                rho=rho,
                reduce="mean",
            )
            train_mon = sigma_monotonicity_loss(sigma_net, N_col_base)
            reg = lambda_reg * sum((p ** 2).sum() for p in sigma_net.parameters())

            train_total = train_nll + lambda_mon * train_mon + reg
            train_total.backward()
            opt.step()

            with torch.no_grad():
                rho_eval = torch.tanh(rho_raw)

                if use_validation:
                    sigma_pred_val = sigma_net(N_val)
                    val_nll = ar1_gaussian_nll_batch(
                        resid_all=r_val,
                        sigma_all=sigma_pred_val,
                        rho=rho_eval,
                        reduce="mean",
                    )
                else:
                    val_nll = torch.tensor(np.nan, device=device)

            val_mon = sigma_monotonicity_loss(sigma_net, N_col_base)
            val_total = (
                val_nll + lambda_mon * val_mon
                if use_validation
                else torch.tensor(np.nan, device=device)
            )

            train_total_history.append(train_total.item())
            train_nll_history.append(train_nll.item())
            train_mon_history.append(train_mon.item())

            val_total_history.append(val_total.item())
            val_nll_history.append(val_nll.item())
            val_mon_history.append(val_mon.item() if use_validation else np.nan)

            if (epoch == 0) or ((epoch + 1) % print_every == 0) or (epoch == n_epochs - 1):
                if use_validation:
                    print(
                        f"{label} | Repeat {repeat + 1:2d}/{n_repeats} | "
                        f"Epoch {epoch + 1:5d}/{n_epochs} | "
                        f"rho={rho_eval.item(): .4f} | "
                        f"Train Total: {train_total.item():.6e} | "
                        f"Train NLL: {train_nll.item():.6e} | "
                        f"Train Mon: {train_mon.item():.6e} | "
                        f"Val Total: {val_total.item():.6e} | "
                        f"Val NLL: {val_nll.item():.6e} | "
                        f"Val Mon: {val_mon.item():.6e}"
                    )
                else:
                    print(
                        f"{label} | Repeat {repeat + 1:2d}/{n_repeats} | "
                        f"Epoch {epoch + 1:5d}/{n_epochs} | "
                        f"rho={rho_eval.item(): .4f} | "
                        f"Train Total: {train_total.item():.6e} | "
                        f"Train NLL: {train_nll.item():.6e} | "
                        f"Train Mon: {train_mon.item():.6e}"
                    )

            criterion = val_total.item() if use_validation else train_total.item()

            if criterion < best_val_loss:
                best_val_loss = criterion
                best_epoch = epoch
                best_state = {
                    "sigma_net": copy.deepcopy(sigma_net.state_dict()),
                    "rho_raw": rho_raw.detach().clone(),
                }

        sigma_net.load_state_dict(best_state["sigma_net"])
        rho_raw_best = best_state["rho_raw"]
        rho_best = torch.tanh(rho_raw_best).item()
        sigma_net.eval()

        with torch.no_grad():
            sigma_curve = sigma_net(N_eval_grid).squeeze(1).cpu().numpy()

        best_models.append(copy.deepcopy(sigma_net))
        train_total_histories.append(train_total_history)
        train_nll_histories.append(train_nll_history)
        train_mon_histories.append(train_mon_history)

        val_total_histories.append(val_total_history)
        val_nll_histories.append(val_nll_history)
        val_mon_histories.append(val_mon_history)

        curve_histories.append(sigma_curve)
        rho_histories.append(rho_best)

        best_val_losses.append(best_val_loss)
        best_epochs.append(best_epoch)
        best_rhos.append(rho_best)

        print(
            f"Finished {label} repeat {repeat + 1:2d}/{n_repeats} | "
            f"Best epoch = {best_epoch:5d} | "
            f"Best loss = {best_val_loss:.6e} | "
            f"Best rho = {rho_best:.4f}"
        )

    return {
        "train_total_histories": np.array(train_total_histories),
        "train_nll_histories": np.array(train_nll_histories),
        "train_mon_histories": np.array(train_mon_histories),
        "val_total_histories": np.array(val_total_histories),
        "val_nll_histories": np.array(val_nll_histories),
        "val_mon_histories": np.array(val_mon_histories),
        "curve_histories": np.array(curve_histories),
        "rho_histories": np.array(rho_histories),
        "best_val_losses": np.array(best_val_losses),
        "best_epochs": np.array(best_epochs),
        "best_rhos": np.array(best_rhos),
        "best_models": best_models,
    }