import numpy as np
from numpy.random import SeedSequence

from utils.validation import validate_option_params, validate_simulation_params

# A seed-like value accepted by numpy.random.default_rng. We deliberately allow a
# SeedSequence (in addition to int/None) so callers can spawn independent child
# streams (e.g. exotics.py) and pass them straight through without touching global state.
SeedLike = int | SeedSequence | None


def simulate_gbm_paths(S0: float, r: float, sigma: float, T: float, n_steps: int, n_paths: int, seed: SeedLike = None, antithetic: bool = False) -> np.ndarray:
    """
    Simulates Geometric Brownian Motion paths.
    Returns array of shape (n_steps + 1, n_paths).

    Randomness uses numpy's modern Generator (np.random.default_rng) so calls are
    reproducible and isolated from global RNG state.
    """
    validate_option_params(S0, 1.0, T, sigma) # K is not used here
    validate_simulation_params(n_steps, n_paths)

    rng = np.random.default_rng(seed)

    dt = T / n_steps
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S0

    if antithetic:
        n_half = n_paths // 2
        Z = rng.standard_normal((n_steps, n_half))
        Z = np.concatenate((Z, -Z), axis=1)
    else:
        Z = rng.standard_normal((n_steps, n_paths))

    for step in range(1, n_steps + 1):
        paths[step] = paths[step-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[step-1])

    return paths

def simulate_gbm_paths_student_t(S0: float, r: float, sigma: float, T: float, n_steps: int, n_paths: int, df: float = 3.0, seed: SeedLike = None, antithetic: bool = False) -> np.ndarray:
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
        paths[step] = paths[step-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[step-1])

    return paths
