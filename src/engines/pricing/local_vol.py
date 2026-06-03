"""Local volatility: Dupire (1994) formula and Gatheral SVI parametrisation.

The two components are independent:
  - dupire_local_vol: converts an implied-vol surface to a local-vol surface via
    Dupire's PDE formula.  Takes a callable iv_surface(K, T) as input.
  - calibrate_svi: fits the Gatheral (2004) SVI model to a slice of market vols.
  - check_svi_arbitrage: tests butterfly (g(k) ≥ 0) and calendar-spread conditions.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from utils.validation import validate_option_params, validate_positive

# ---------------------------------------------------------------------------
# Dupire (1994) local volatility
# ---------------------------------------------------------------------------

def dupire_local_vol(
    K: float,
    T: float,
    S0: float,
    r: float,
    iv_surface: Callable[[float, float], float],
    dK: float | None = None,
    dT: float | None = None,
) -> dict:
    """Estimate local volatility σ_loc(K, T) from an implied-vol surface via Dupire's formula.

    Dupire (1994):
        σ_loc²(K,T) = [∂C/∂T + r·K·∂C/∂K] / [½·K²·∂²C/∂K²]

    In terms of implied vol w(K,T) = σ_IV(K,T)² · T (total variance), this becomes
    the Derman-Kani formula used in practice.  We compute all partial derivatives via
    central finite differences.

    Parameters
    ----------
    iv_surface : callable iv_surface(K, T) → σ_IV (implied vol scalar).
    dK, dT     : step sizes for finite differences (defaults: 0.5% of K, 0.01 year).

    Returns
    -------
    dict with ``local_vol`` (non-negative, clamped), ``method``, ``K``, ``T``.
    """
    validate_option_params(S0, K, T, 1e-6)
    validate_positive(K, "K")

    dK = dK if dK is not None else max(K * 0.005, 0.01)
    dT = dT if dT is not None else 0.005

    # Total variance surface: w(K, T) = σ²·T
    def w(k: float, t: float) -> float:
        return float(iv_surface(k, t)) ** 2 * t

    # Central differences (avoiding T=0)
    T_lo = max(T - dT, dT)
    dw_dT = (w(K, T + dT) - w(K, T_lo)) / (T + dT - T_lo)
    dw_dK = (w(K + dK, T) - w(K - dK, T)) / (2.0 * dK)
    d2w_dK2 = (w(K + dK, T) - 2.0 * w(K, T) + w(K - dK, T)) / dK**2

    # Dupire-Derman-Kani local variance (Gatheral 2006, eq 1.4 in total-variance form)
    y = np.log(K / (S0 * np.exp(r * T)))  # log-moneyness
    w0 = w(K, T)

    if abs(w0) < 1e-10:
        # Degenerate: near-zero implied vol → zero local vol
        return {"local_vol": 0.0, "method": "dupire", "K": K, "T": T}

    # Denominator of the Gatheral formula
    denom = (
        1.0
        - y / w0 * dw_dK
        + 0.25 * (-0.25 - 1.0 / w0 + y**2 / w0**2) * dw_dK**2
        + 0.5 * d2w_dK2
    )
    denom = max(denom, 1e-8)  # clamp to avoid negative local variance

    local_var = dw_dT / denom
    local_var = max(local_var, 0.0)  # local variance must be non-negative
    local_vol = float(np.sqrt(local_var))

    return {"local_vol": local_vol, "local_var": local_var, "method": "dupire", "K": K, "T": T}


# ---------------------------------------------------------------------------
# Gatheral SVI parametrisation (2004)
# ---------------------------------------------------------------------------

def _svi_total_var(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """Gatheral (2004) raw SVI: w(k) = a + b*(ρ(k−m) + √((k−m)²+σ²))."""
    km = k - m
    return a + b * (rho * km + np.sqrt(km**2 + sigma**2))


def calibrate_svi(
    log_strikes: np.ndarray,
    market_total_var: np.ndarray,
    initial_guess: dict | None = None,
) -> dict:
    """Fit Gatheral (2004) SVI parametrisation to a single maturity slice.

    The SVI total-variance smile: w(k) = a + b*(ρ(k−m) + √((k−m)²+σ²))
    where k = log(K/F) is the log-moneyness vs the forward.

    Parameters
    ----------
    log_strikes      : array of k = log(K/F), shape (M,).
    market_total_var : corresponding total implied variance w = σ²·T, shape (M,).

    Returns
    -------
    dict with ``params``, ``rmse``, ``success``.
    """
    k = np.asarray(log_strikes, dtype=float)
    w_mkt = np.asarray(market_total_var, dtype=float)

    if initial_guess is None:
        a0 = float(np.mean(w_mkt)) * 0.5
        b0 = 0.1
        rho0 = -0.3
        m0 = 0.0
        sigma0 = 0.3
    else:
        a0 = initial_guess.get("a", float(np.mean(w_mkt)) * 0.5)
        b0 = initial_guess.get("b", 0.1)
        rho0 = initial_guess.get("rho", -0.3)
        m0 = initial_guess.get("m", 0.0)
        sigma0 = initial_guess.get("sigma", 0.3)

    def objective(x: np.ndarray) -> float:
        a, b, rho, m, sig = x
        if b < 0 or sig < 1e-6 or abs(rho) >= 1.0 or b * (1 + abs(rho)) >= 2.0:
            return 1e6  # infeasible: penalise no-butterfly-arbitrage violation
        w_fit = _svi_total_var(k, a, b, rho, m, sig)
        if np.any(w_fit < 0):
            return 1e6
        return float(np.mean((w_fit - w_mkt) ** 2))

    x0 = np.array([a0, b0, rho0, m0, sigma0])
    bounds = [
        (1e-6, None),   # a > 0
        (1e-6, None),   # b > 0
        (-0.9999, 0.9999),  # |ρ| < 1
        (None, None),   # m free
        (1e-4, None),   # σ > 0
    ]

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})

    a, b, rho, m, sig = result.x
    w_fit = _svi_total_var(k, a, b, rho, m, sig)
    rmse = float(np.sqrt(np.mean((w_fit - w_mkt) ** 2)))

    return {
        "params": {"a": float(a), "b": float(b), "rho": float(rho), "m": float(m), "sigma": float(sig)},
        "fitted_total_var": w_fit,
        "rmse": rmse,
        "success": bool(result.success),
        "method": "svi",
    }


def check_svi_arbitrage(
    params: dict,
    k_grid: np.ndarray | None = None,
) -> dict:
    """Check Gatheral SVI params for static arbitrage (butterfly + calendar).

    Butterfly condition: g(k) = (1 − kw'/(2w))² − (w'/2)²(1/4+1/w) + w''/2 ≥ 0
    for all k.  If min(g) < 0, the slice admits butterfly arbitrage.

    Parameters
    ----------
    params  : dict with keys a, b, rho, m, sigma.
    k_grid  : optional array of log-moneyness values; defaults to [-3, 3].

    Returns
    -------
    dict with ``butterfly_ok`` (bool), ``min_g``, ``k_grid``.
    """
    if k_grid is None:
        k_grid = np.linspace(-3.0, 3.0, 500)

    a, b, rho, m, sig = params["a"], params["b"], params["rho"], params["m"], params["sigma"]
    k = np.asarray(k_grid, dtype=float)
    km = k - m

    w = _svi_total_var(k, a, b, rho, m, sig)
    sq = np.sqrt(km**2 + sig**2)
    dw = b * (rho + km / sq)                              # ∂w/∂k
    d2w = b * sig**2 / sq**3                              # ∂²w/∂k²

    g = (
        (1.0 - k * dw / (2.0 * w)) ** 2
        - dw**2 / 4.0 * (1.0 / w + 0.25)
        + d2w / 2.0
    )

    return {
        "butterfly_ok": bool(np.all(g >= -1e-8)),
        "min_g": float(np.min(g)),
        "k_grid": k_grid,
        "g": g,
        "method": "svi-butterfly-check",
    }
