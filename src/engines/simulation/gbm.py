import numpy as np
from numpy.random import SeedSequence

from utils.validation import validate_option_params, validate_simulation_params

# A seed-like value accepted by numpy.random.default_rng. We deliberately allow a
# SeedSequence (in addition to int/None) so callers can spawn independent child
# streams (e.g. exotics.py) and pass them straight through without touching global state.
SeedLike = int | SeedSequence | None


def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: SeedLike = None,
    antithetic: bool = False,
    quasi_mc: bool = False,
    moment_matching: bool = False,
) -> np.ndarray:
    """
    Simulates Geometric Brownian Motion paths.
    Returns array of shape (n_steps + 1, n_paths).

    Parameters
    ----------
    quasi_mc:
        If True, use Sobol low-discrepancy normals instead of pseudo-random draws.
    moment_matching:
        If True, adjust Z so each time-step has sample mean=0 and std=1.
    """
    validate_option_params(S0, 1.0, T, sigma)  # K is not used here
    validate_simulation_params(n_steps, n_paths)

    dt = T / n_steps
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S0

    if quasi_mc:
        from engines.simulation.variance_reduction import sobol_standard_normal

        n_half = n_paths // 2
        _seed = seed if isinstance(seed, int) else 0
        if antithetic:
            z_raw = sobol_standard_normal(n_dims=n_steps, n_points=n_half, seed=_seed)
            Z = np.concatenate((z_raw.T, -z_raw.T), axis=1)
        else:
            z_raw = sobol_standard_normal(n_dims=n_steps, n_points=n_paths, seed=_seed)
            Z = z_raw.T
    else:
        rng = np.random.default_rng(seed)
        if antithetic:
            n_half = n_paths // 2
            z_half = rng.standard_normal((n_steps, n_half))
            Z = np.concatenate((z_half, -z_half), axis=1)
        else:
            Z = rng.standard_normal((n_steps, n_paths))

    if moment_matching:
        from engines.simulation.variance_reduction import apply_moment_matching

        Z = apply_moment_matching(Z)

    for step in range(1, n_steps + 1):
        paths[step] = paths[step - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[step - 1])

    return paths


def simulate_gbm_paths_student_t(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    df: float = 3.0,
    seed: SeedLike = None,
    antithetic: bool = False,
) -> np.ndarray:
    """
    Simulates Geometric Brownian Motion paths using Student-t innovations for stress testing.
    Returns array of shape (n_steps + 1, n_paths).
    """
    rng = np.random.default_rng(seed)

    dt = T / n_steps
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S0

    # Scale factor to match variance of standard normal if df > 2
    # Variance of t-dist is df / (df - 2)
    scale = np.sqrt((df - 2) / df) if df > 2 else 1.0

    if antithetic:
        n_half = n_paths // 2
        Z = rng.standard_t(df, size=(n_steps, n_half)) * scale
        Z = np.concatenate((Z, -Z), axis=1)
    else:
        Z = rng.standard_t(df, size=(n_steps, n_paths)) * scale

    for step in range(1, n_steps + 1):
        paths[step] = paths[step - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[step - 1])

    return paths
