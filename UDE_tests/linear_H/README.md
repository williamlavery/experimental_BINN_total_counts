# PINN AR(1) logistic example with factored dynamics and linear H

This package is the like-for-like version of the factored codebase for the parametrization

- original factored form: `dN/dt = N * H(N)`
- enforced model class here: `H(N) = a * N + b`
- learned dynamics: `dN/dt = N * (a * N + b)`

The data generation is still logistic growth, so the ground-truth `H_true(N) = r_true * (1 - N)` is already linear. The learned dynamics model now enforces that `H(N)` is linear instead of representing it with a nonlinear neural network.

## Files

- `config.py` - experiment settings (unchanged interface)
- `data.py` - synthetic data generation and RK4 solver using `H_true` and `G_true = N * H_true`
- `models.py` - `DynamicsNet` is now a single linear layer implementing `H(N) = a*N + b`
- `losses.py` - unchanged PINN residual `dN/dt - N*H(N)` and `h_zero_nonnegative_penalty`
- `training.py` - unchanged training pipeline; uses the linear `DynamicsNet` automatically
- `diagnostics.py` - unchanged plotting helpers
- `utils.py` - unchanged utilities
- `notebook_run_linear_H.ipynb` - notebook analogous to the factored notebook, updated for linear `H`

## Notes

- This is intended to be a drop-in analogous version of the uploaded modularized script set.
- Output names such as `rhs_histories` are preserved for compatibility, but they now store the learned linear `H(N)` curve on `N_grid`.
- `hidden_dim_dyn` is kept in the config for interface compatibility, but it is not used by the linear `DynamicsNet`.
