# rPINN AR(1) logistic example

This package now uses a randomized PINN (rPINN-style) training loop for the logistic-growth inverse problem with heteroscedastic AR(1) observation noise.

## Files

- `config.py` - experiment settings, including fixed `sigma_data_rpinn=0.1` and `sigma_phys_rpinn=0.1`
- `data.py` - synthetic data generation and RK4 solver
- `models.py` - neural network models
- `losses.py` - Gaussian and AR(1) likelihoods, PINN losses, and randomized residual loss
- `training.py` - rPINN training and sigma/rho fitting
- `diagnostics.py` - plotting helpers
- `notebook_run.ipynb` - runnable notebook

## What changed

- Replaced the deterministic multi-IC PINN training loop with an rPINN-style ensemble solver.
- Each ensemble member solves a randomized objective with fixed loss uncertainties:
  - data-loss sigma = `0.1`
  - physics-loss sigma = `0.1`
- Observation randomization is fixed per ensemble member.
- Physics randomization is fixed on a per-member collocation bank and minibatched during optimization.
- The old `fit_multi_ic_pinn(...)` entry point remains as a backward-compatible alias to the new rPINN trainer.

## Notes

- The sigma/rho fitting utilities are still present and can be run on the posterior-mean observation fits produced by the rPINN ensemble.
- This is an rPINN-style implementation adapted to the original code structure, not a verbatim reproduction of the paper's full Bayesian machinery.
