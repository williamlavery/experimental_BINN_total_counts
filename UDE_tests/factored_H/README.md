# PINN AR(1) logistic example with factored dynamics

This package is the like-for-like version of the original codebase for the parametrization

- original: `dN/dt = G(N)`
- factored: `dN/dt = N * H(N)`

The data generation is still logistic growth, but the learned dynamics network now represents `H(N)` rather than the full right-hand side `G(N)`.

## Files

- `config.py` - experiment settings (`lambda_h0` replaces `lambda_g0`)
- `data.py` - synthetic data generation and RK4 solver using `H_true` and `G_true = N * H_true`
- `models.py` - unchanged network definitions; `DynamicsNet` now learns `H(N)`
- `losses.py` - PINN residual uses `N * H(N)` and the boundary penalty is `h_zero_nonnegative_penalty`
- `training.py` - rollout and physics loss use `N * dyn_net(N)` throughout
- `diagnostics.py` - unchanged plotting helpers
- `utils.py` - unchanged utilities

## Notes

- This is intended to be a drop-in analogous version of the original modularized script.
- Output names such as `rhs_histories` are preserved for compatibility, but they now store the learned `H(N)` curve on `N_grid`.
