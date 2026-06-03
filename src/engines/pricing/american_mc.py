"""Longstaff-Schwartz (2001) American option pricing via Monte Carlo least-squares.

The algorithm (LSM) prices American options by backward induction:
  1. Simulate paths to maturity.
  2. At each exercise date (working backward), regress the discounted continuation
     value on in-the-money paths using a polynomial basis.
  3. Exercise where immediate payoff ≥ fitted continuation value.

Key properties:
  - Low-bias estimator: the exercise strategy is sub-optimal, so the price is a
    lower bound on the true American price.
  - Converges to the true price as n_paths → ∞ and the basis spans the payoff space.
"""
from __future__ import annotations

import numpy as np

from engines.simulation.gbm import simulate_gbm_paths
from utils.validation import validate_non_negative, validate_option_params, validate_positive


def american_option_lsm(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
    n_steps: int = 100,
    n_paths: int = 10_000,
    poly_degree: int = 3,
    seed: int | np.random.SeedSequence | None = None,
) -> dict:
    """Price an American option using Longstaff-Schwartz (2001) least-squares Monte Carlo.

    Parameters
    ----------
    poly_degree : degree of the Laguerre / monomial basis for the continuation-value
                  regression.  3 (cubic) is sufficient for most cases.

    Returns
    -------
    dict with keys:
        price         : float — LSM price (low-bias estimator).
        std_err       : float — standard error of the MC estimate.
        ci_low/hi     : float — 95% confidence interval bounds.
        exercise_frac : float — fraction of paths where early exercise occurred.
        exercise_boundary : ndarray, shape (n_steps,) — estimated exercise boundary
                            at each step (np.nan when no in-the-money paths).
        method        : "lsm"
    """
    validate_option_params(S0, K, T, sigma)
    validate_non_negative(r, "r")
    validate_positive(n_steps, "n_steps")
    validate_positive(n_paths, "n_paths")

    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)
    # paths: (n_steps+1, n_paths);  paths[0] = S0

    dt = T / n_steps
    discount = np.exp(-r * dt)

    def payoff(S: np.ndarray) -> np.ndarray:
        if is_call:
            return np.maximum(S - K, 0.0)
        return np.maximum(K - S, 0.0)

    # Initialise cashflow matrix: cashflow[t, i] = discounted cash at step t for path i
    # We track a cashflow vector: the discounted value at each path's optimal exercise time.
    cash_flow = payoff(paths[-1])  # shape (n_paths,) — terminal payoff

    exercise_boundary = np.full(n_steps, np.nan)

    # Backward induction over exercise dates
    for t in range(n_steps - 1, 0, -1):
        S_t = paths[t]
        imm = payoff(S_t)

        itm = imm > 0.0  # in-the-money paths for regression
        if itm.sum() < poly_degree + 2:
            # Too few ITM paths to fit; carry cash flow forward
            cash_flow *= discount
            continue

        S_itm = S_t[itm]
        y_itm = cash_flow[itm] * discount  # discounted continuation

        # Polynomial / Laguerre basis on normalised spot
        X = _make_basis(S_itm / K, poly_degree)
        coeffs, _, _, _ = np.linalg.lstsq(X, y_itm, rcond=None)
        continuation = X @ coeffs

        ex = imm[itm] >= continuation
        if ex.any():
            exercise_boundary[t] = float(S_itm[ex].mean())

        # Where immediate payoff ≥ estimated continuation: exercise
        cash_flow[itm] = np.where(ex, imm[itm], y_itm)
        cash_flow[~itm] *= discount  # non-ITM paths just carry forward

    # Discount all remaining cash flows to t=0
    price_paths = cash_flow * discount  # one more step to t=0

    price = float(np.mean(price_paths))
    std = float(np.std(price_paths, ddof=1) / np.sqrt(n_paths))
    exercise_frac = float(np.mean(payoff(paths[1:-1]) > 0.0))  # rough fraction ITM

    return {
        "price": price,
        "std_err": std,
        "ci_low": price - 1.96 * std,
        "ci_hi": price + 1.96 * std,
        "exercise_frac": exercise_frac,
        "exercise_boundary": exercise_boundary,
        "method": "lsm",
        "settings": {
            "n_steps": n_steps,
            "n_paths": n_paths,
            "poly_degree": poly_degree,
            "is_call": is_call,
        },
    }


def _make_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """Build Laguerre-inspired polynomial basis: columns [1, x, x², ..., x^degree].

    For American option regression, monomials work well for moderate degree.
    """
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x**d)
    return np.column_stack(cols)
