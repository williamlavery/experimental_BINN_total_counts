# Updated scripts

This bundle updates the training flow so that rPINN uses the learned noise/sigma model to build the data uncertainty, and uses that same sigma model for the physics term as well.

## Main change

`fit_multi_ic_rpinn(...)` and `fit_multi_ic_pinn(...)` now accept either:

- `sigma_fit_result=<output of fit_sigma_ar1_repeated(...)>`
- `sigma_model=<single SigmaNet or list of SigmaNet models>`

When a sigma model is supplied, the training code:

- evaluates the sigma model for the **data term**
- evaluates the same sigma model for the **physics term**
- falls back to the old constant sigmas only if no sigma model is supplied

## Example

```python
sigma_result = fit_sigma_ar1_repeated(...)

rpinn_result = fit_multi_ic_rpinn(
    cfg=cfg,
    y_data_all=data_dict["y_data_all"],
    t_obs=data_dict["t_obs"],
    t_plot=data_dict["t_plot"],
    N_grid=data_dict["N_grid"],
    n_ics=len(cfg.N0_list),
    device=device,
    sigma_fit_result=sigma_result,
)
```

## Config knobs

The updated `ExperimentConfig` includes:

- `use_learned_sigma_rpinn=True`
- `sigma_eval_mode="predicted_state"`
- `sigma_detach_state=True`
- `sigma_floor_rpinn=1e-4`

`predicted_state` means sigma is evaluated at the current PINN-predicted state.
`observed_state` can be selected if you prefer to evaluate the data sigma on the noisy observation values instead.
