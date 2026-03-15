# PINN AR(1) logistic example

This package contains a modularized version of the logistic-growth PINN with heteroscedastic AR(1) observation noise.

## Files

- `config.py` - experiment settings
- `data.py` - synthetic data generation and RK4 solver
- `models.py` - neural network models
- `losses.py` - Gaussian and AR(1) likelihoods plus PINN losses
- `training.py` - PINN training and sigma/rho fitting
- `diagnostics.py` - plotting helpers
- `notebook_run.ipynb` - runnable notebook

## Notes

- The original script was not a segmentation model; it is a PINN for logistic growth.
- I corrected undefined variables and inconsistent sigma labels.
- The notebook is set up to run from the same folder as these modules.
