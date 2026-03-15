# rPINN AR(1) logistic example

This package uses a randomized PINN (rPINN-style) training loop for the logistic-growth inverse problem with heteroscedastic AR(1) observation noise.

The learned dynamics are parameterized as

\[
G(N) = N\,H(N)
\]

so the network is inspected both through the reduced factor \(H(N)\) and through the full RHS \(G(N)\).

## Files

- `config.py` - experiment settings
- `data.py` - synthetic data generation, RK4 solver, and true `H(N)` / `G(N)` helpers
- `models.py` - neural network models; `DynamicsNet.h(N)` returns the learned `H(N)` and `forward(N)` returns the full `G(N)`
- `losses.py` - Gaussian and AR(1) likelihoods, PINN residuals, and randomized residual loss
- `training.py` - rPINN training and sigma/rho fitting
- `diagnostics.py` - plotting helpers
- `notebook_run.ipynb` - runnable notebook

## What changed

- The dynamics network now enforces the structural factorization `G(N) = N * H(N)`.
- The synthetic truth is written analogously with `H_true(N) = r * (1 - N)` and `G_true(N) = N * H_true(N)`.
- The physics residual is still `dN/dt - G(N)`, so the PINN loss stays analogous.
- The old `G(0)` penalty was removed because `G(0)=0` is now enforced by construction.
- Training returns diagnostics for both objects:
  - learned / MAP `H(N)` via `map_h`, `h_histories`, `h_mean`, `h_lo`, `h_hi`
  - learned / MAP `G(N)` via `map_rhs`, `rhs_histories`, `rhs_mean`, `rhs_lo`, `rhs_hi`

## Notes

- The old `fit_multi_ic_pinn(...)` entry point remains as a backward-compatible alias to the rPINN trainer.
- To inspect the structural benefit of the parameterization, compare both `H(N)` vs `H_true(N)` and `N*H(N)` vs `G_true(N)` in the notebook.
