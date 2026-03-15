from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    seed_torch: int = 0
    seed_numpy: int = 0
    device: str = "cpu"

    # Ground-truth dynamics
    r_true: float = 1.0
    N0_list: tuple = (0.10, 0.12, 0.13, 0.09)

    # Time grids
    t0: float = 0.0
    t1: float = 6.0
    numpts: int = 24
    n_plot: int = 200
    n_grid: int = 200

    # Noise
    noise_sig_coeff: float = 0.1
    gamma: float = 1.0
    rho_true: float = 0.5

    # rPINN / randomized-PINN setup
    n_col: int = 1000
    n_epochs_pinn: int = 2000
    val_every: int = 100
    print_every_pinn: int = 500
    batch_obs: int = 24
    batch_col: int = 100
    lambda_phys: float = 1.0
    lambda_ic: float = 0.0
    learning_rate_pinn: float = 1e-3
    hidden_dim_sol: int = 32
    hidden_dim_dyn: int = 32
    factor_rhs_by_state: bool = True
    dyn_init_seed_base: int = 10000
    sol_init_seed_base: int = 20000

    # Fixed loss uncertainties requested by user
    sigma_data_rpinn: float = 0.1
    sigma_phys_rpinn: float = 0.1

    # Number of randomized inverse solves / posterior particles
    n_rpinn_samples: int = 8

    # Sigma / AR1 fitting
    n_repeats_sigma: int = 1
    val_fraction_sigma: float = 0.2
    n_epochs_sigma: int = 2000
    lr_sigma: float = 5e-3
    hidden_dim_sigma: int = 8
    lambda_mon: float = 1.0
    lambda_reg: float = 1e-5
    sigma_split_seed_base_1: int = 5000
    sigma_model_seed_base_1: int = 7000
    sigma_split_seed_base_2: int = 9000
    sigma_model_seed_base_2: int = 11000
    print_every_sigma: int = 250
