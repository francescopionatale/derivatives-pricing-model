"""Variance reduction techniques for Monte Carlo simulation.

Provides:
- mc_with_control_variate: generic beta* control variate estimator
- sobol_standard_normal: low-discrepancy Sobol normals via scipy.stats.qmc
- apply_moment_matching: per-step mean/std normalisation
"""

from collections.abc import Callable

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import Sobol


def mc_with_control_variate(
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    control_payoff_fn: Callable[[np.ndarray], np.ndarray],
    control_exact: float,
    paths: np.ndarray,
    r: float,
    T: float,
) -> dict:
    """Reduce Monte Carlo variance using a control variate.

    Parameters
    ----------
    payoff_fn:
        Callable that maps paths (n_steps+1, n_paths) to a 1-D array of
        *discounted* per-path payoffs (shape (n_paths,)).
    control_payoff_fn:
        Same signature — discounted per-path payoffs of the control instrument.
    control_exact:
        Known analytical price of the control instrument (used to centre X).
    paths:
        GBM paths array of shape (n_steps+1, n_paths).
    r:
        Risk-free rate (passed for reference; discounting is caller's
        responsibility inside payoff_fn / control_payoff_fn).
    T:
        Time to maturity (same note as r).

    Returns
    -------
    dict with keys:
        price                  – control-variate adjusted price
        std_err                – standard error of the adjusted estimator
        beta                   – optimal beta* coefficient
        variance_reduction_ratio – Var(Y) / Var(Y_cv)
        correlation            – sample correlation between Y and X
    """
    # Validate inputs
    if paths.ndim != 2:
        raise ValueError(f"paths must be 2-D, got shape {paths.shape}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    Y = np.asarray(payoff_fn(paths), dtype=float)
    X = np.asarray(control_payoff_fn(paths), dtype=float)

    n_paths = Y.shape[0]
    if n_paths < 2:
        raise ValueError("Need at least 2 paths to estimate variance.")

    var_X = np.var(X, ddof=1)
    if var_X == 0.0:
        raise ValueError("Control variate has zero variance — cannot estimate beta.")

    cov_YX = np.cov(Y, X, ddof=1)[0, 1]
    beta_star = -cov_YX / var_X

    # Y_cv = Y + beta_star * (X - control_exact)
    # With beta_star = -Cov(Y,X)/Var(X) this minimises Var(Y_cv).
    # price_cv = mean(Y) + beta_star * (mean(X) - control_exact)  [spec eq.]
    Y_cv = Y + beta_star * (X - control_exact)

    price_cv = float(np.mean(Y_cv))
    std_err_cv = float(np.std(Y_cv, ddof=1) / np.sqrt(n_paths))

    var_Y = float(np.var(Y, ddof=1))
    var_Y_cv = float(np.var(Y_cv, ddof=1))
    vrr = var_Y / var_Y_cv if var_Y_cv > 0 else float("inf")

    corr_matrix = np.corrcoef(Y, X)
    correlation = float(corr_matrix[0, 1])

    return {
        "price": price_cv,
        "std_err": std_err_cv,
        "beta": float(beta_star),
        "variance_reduction_ratio": vrr,
        "correlation": correlation,
    }


def sobol_standard_normal(n_dims: int, n_points: int, seed: int = 0) -> np.ndarray:
    """Generate quasi-random standard-normal samples using a Sobol sequence.

    Parameters
    ----------
    n_dims:
        Number of dimensions (e.g. number of time steps).
    n_points:
        Number of sample points (e.g. number of paths or half-paths).
    seed:
        Integer seed for scrambling.

    Returns
    -------
    ndarray of shape (n_points, n_dims) drawn from N(0,1).
    """
    if n_dims < 1:
        raise ValueError(f"n_dims must be >= 1, got {n_dims}")
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")

    engine = Sobol(d=n_dims, scramble=True, seed=seed)
    u = engine.random(n_points)  # shape (n_points, n_dims), uniform [0, 1)

    # Clip away boundaries to avoid ±inf after ppf
    u = np.clip(u, 1e-10, 1 - 1e-10)
    z = norm.ppf(u)

    return z


def apply_moment_matching(Z: np.ndarray) -> np.ndarray:
    """Normalise a random-number array so each time-step has mean=0, std=1.

    Parameters
    ----------
    Z:
        Array of shape (n_steps, n_paths).

    Returns
    -------
    Normalised array of the same shape.
    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-D, got shape {Z.shape}")

    mu = Z.mean(axis=1, keepdims=True)
    sigma = Z.std(axis=1, keepdims=True, ddof=0)

    # Avoid division by zero for degenerate rows
    sigma = np.where(sigma == 0, 1.0, sigma)

    return (Z - mu) / sigma
