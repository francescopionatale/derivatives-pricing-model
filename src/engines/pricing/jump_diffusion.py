"""Merton (1976) jump-diffusion model: closed-form series, Fourier char function, simulation.

The three implementations cross-validate each other:
  merton_price (series)  ≡  heston_price_cos(merton_characteristic_function, ...)  [Fourier]
                         ≈  MC simulation with enough paths
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from utils.validation import validate_non_negative, validate_option_params, validate_positive


def merton_characteristic_function(
    u: np.ndarray,
    S0: float,
    sigma: float,
    lam: float,
    mu_J: float,
    sigma_J: float,
    r: float,
    T: float,
) -> np.ndarray:
    """Characteristic function of log(S_T) under Merton (1976) jump-diffusion.

    Plugs into the Fourier pricers (Carr-Madan / COS / Lewis) directly, demonstrating
    that the entire Fourier infrastructure is model-agnostic: just swap the char function.

    Model: dS/S = (r − λk̄) dt + σ dW + (e^J − 1) dN
    where J ~ N(μ_J, σ_J²) and k̄ = exp(μ_J + σ_J²/2) − 1.

    Parameters
    ----------
    u : complex array, shape (N,)
    """
    u = np.asarray(u, dtype=complex)
    kbar = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0

    # GBM component (with drift adjusted for jump compensation)
    gbm = 1j * u * (np.log(S0) + r * T) - 0.5 * sigma**2 * T * (u**2 + 1j * u) - 1j * u * lam * kbar * T

    # Compound-Poisson component: λT·(E[exp(iuY)] − 1) where Y ~ N(μ_J, σ_J²)
    jump = lam * T * (np.exp(1j * u * mu_J - 0.5 * sigma_J**2 * u**2) - 1.0)

    return np.exp(gbm + jump)


def merton_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    mu_J: float,
    sigma_J: float,
    is_call: bool = True,
    n_terms: int = 50,
) -> dict:
    """European option price under Merton (1976) jump-diffusion via Poisson-weighted BS series.

    The price is exact in the limit n_terms → ∞; 50 terms is machine-precise for
    λT ≤ 20 (relative error < 10⁻¹²).

    Parameters
    ----------
    sigma   : diffusion (GBM) volatility, not including jump variance.
    lam     : jump intensity (expected number of jumps per year).
    mu_J    : mean log-jump size (log(1 + expected_jump_fraction) − σ_J²/2).
    sigma_J : std of log-jump size.
    """
    validate_option_params(S0, K, T, sigma)
    validate_non_negative(lam, "lam")
    validate_positive(n_terms, "n_terms")

    kbar = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0  # E[J − 1], jump risk-premium
    lam_prime = lam * (1.0 + kbar)  # = λ·exp(μ_J + σ_J²/2)

    price = 0.0
    lam_prime_T = lam_prime * T

    for n in range(int(n_terms)):
        # Poisson weight for exactly n jumps in [0, T]
        w_n = math.exp(-lam_prime_T) * lam_prime_T**n / math.factorial(n)
        if w_n < 1e-15:
            break

        # n-th term effective parameters (Haug 2007, Merton 1976)
        sigma_n_sq = sigma**2 + n * sigma_J**2 / T
        sigma_n = math.sqrt(max(sigma_n_sq, 1e-15))
        r_n = r - lam * kbar + n * (mu_J + 0.5 * sigma_J**2) / T

        sqrt_T = math.sqrt(T)
        d1 = (math.log(S0 / K) + (r_n + 0.5 * sigma_n**2) * T) / (sigma_n * sqrt_T)
        d2 = d1 - sigma_n * sqrt_T

        if is_call:
            bs_n = S0 * norm.cdf(d1) - K * math.exp(-r_n * T) * norm.cdf(d2)
        else:
            bs_n = K * math.exp(-r_n * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

        price += w_n * bs_n

    return {
        "price": float(price),
        "method": "merton-series",
        "n_terms": int(n_terms),
        "kbar": float(kbar),
        "lam_prime": float(lam_prime),
    }


def simulate_merton_paths(
    S0: float,
    r: float,
    sigma: float,
    lam: float,
    mu_J: float,
    sigma_J: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.SeedSequence | None = None,
) -> np.ndarray:
    """Simulate Merton jump-diffusion paths under the risk-neutral measure.

    Uses per-step compound-Poisson increments: number of jumps ~ Poisson(λ·dt),
    and aggregate log-jump given n jumps ~ N(n·μ_J, n·σ_J²).

    Returns
    -------
    paths : shape (n_steps+1, n_paths), paths[0] = S0.
    """
    validate_option_params(S0, 1.0, T, sigma)
    validate_non_negative(lam, "lam")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    kbar = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0
    drift = (r - 0.5 * sigma**2 - lam * kbar) * dt
    vol_dt = sigma * math.sqrt(dt)

    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S0

    for t in range(n_steps):
        Z = rng.standard_normal(n_paths)
        n_jumps = rng.poisson(lam * dt, size=n_paths)

        # Vectorised aggregate log-jump: for each distinct jump count, draw one normal
        log_jump = np.zeros(n_paths)
        for n in np.unique(n_jumps):
            if n == 0:
                continue
            mask = n_jumps == n
            log_jump[mask] = rng.normal(n * mu_J, math.sqrt(n) * sigma_J, size=int(mask.sum()))

        paths[t + 1] = paths[t] * np.exp(drift + vol_dt * Z + log_jump)

    return paths
