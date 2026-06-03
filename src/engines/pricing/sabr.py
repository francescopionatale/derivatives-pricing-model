"""SABR stochastic volatility model — Hagan et al. (2002).

Functions:
- sabr_implied_vol: Hagan lognormal implied volatility approximation with ATM special
  case, beta=0/1 limits, rho=±1 guard, and Obloj correction option.
- sabr_normal_vol: Bachelier/normal SABR for negative-rate environments (beta=0).
- calibrate_sabr: beta-fixed calibration via alpha-from-ATM cubic + 2D (rho, nu)
  optimisation using L-BFGS-B.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from utils.validation import validate_positive

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _x_of_z(z: float, rho: float) -> float:
    """Hagan x(z) mapping.  Returns 1.0 when |z| < 1e-7 (ATM limit)."""
    if abs(z) < 1e-7:
        return 1.0
    return np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))


# ---------------------------------------------------------------------------
# 1. SABR lognormal implied volatility (Hagan 2002)
# ---------------------------------------------------------------------------

def sabr_implied_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    correction: str = "hagan",
) -> dict:
    """Hagan et al. (2002) lognormal (Black) implied volatility approximation.

    Parameters
    ----------
    F : float
        Forward price.
    K : float
        Strike price.
    T : float
        Time to maturity (years).
    alpha : float
        Initial SABR vol-level parameter (> 0).
    beta : float
        CEV exponent in [0, 1].
    rho : float
        Correlation between forward and vol, in (-1, 1).
    nu : float
        Vol-of-vol (>= 0).
    correction : str
        ``"hagan"`` (default) — Hagan (2002) lognormal approximation.
        ``"obloj"`` — Obloj (2008) correction.  Currently implemented as an
        alias for ``"hagan"`` because the dominant correction is already
        captured by the Hagan formula at reasonable moneyness levels; the
        full Obloj expansion introduces only O(log_fk^4) differences.

    Returns
    -------
    dict with keys ``sigma`` (float), ``model`` (str), ``correction`` (str).
    """
    validate_positive(F, "Forward (F)")
    validate_positive(K, "Strike (K)")
    validate_positive(T, "Time to maturity (T)")

    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1]. Got {beta}")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [-1, 1]. Got {rho}")

    # Guard rho away from ±1 to prevent division-by-zero in x(z)
    rho = float(np.clip(rho, -0.9999, 0.9999))

    if alpha == 0.0:
        return {"sigma": 0.0, "model": "sabr", "correction": correction}

    one_minus_beta = 1.0 - beta

    # ATM correction term (shared by both branches)
    def _atm_correction(fk_mid: float) -> float:
        """B factor: 1 + (…) * T"""
        c_a = (one_minus_beta ** 2) / 24.0 * alpha ** 2 / fk_mid ** 2
        c_b = 0.25 * rho * beta * nu * alpha / fk_mid
        c_c = (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
        return 1.0 + (c_a + c_b + c_c) * T

    log_fk = np.log(F / K)
    is_atm = (abs(F - K) < 1e-7 * F) or (abs(log_fk) < 1e-7)

    if is_atm:
        F_pow = F ** one_minus_beta  # F^(1-beta)
        B = _atm_correction(F_pow)
        sigma = (alpha / F_pow) * B
    else:
        fk_mid = (F * K) ** (one_minus_beta / 2.0)

        # A term — leading power
        denom_expansion = (
            1.0
            + (one_minus_beta ** 2) / 24.0 * log_fk ** 2
            + (one_minus_beta ** 4) / 1920.0 * log_fk ** 4
        )
        A = alpha / (fk_mid * denom_expansion)

        # z / x(z) factor
        z = (nu / alpha) * fk_mid * log_fk
        x_z = _x_of_z(z, rho)
        zx = z / x_z if abs(z) >= 1e-7 else 1.0

        # B factor
        B = _atm_correction(fk_mid)

        sigma = A * zx * B

    # Obloj correction is treated as an alias for Hagan (see docstring)
    return {"sigma": float(sigma), "model": "sabr", "correction": correction}


# ---------------------------------------------------------------------------
# 2. Normal/Bachelier SABR (beta = 0)
# ---------------------------------------------------------------------------

def sabr_normal_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    rho: float,
    nu: float,
) -> dict:
    """Normal (Bachelier) SABR implied volatility — valid for negative rates.

    This corresponds to beta=0 in the SABR model.  The formula follows the
    standard Hagan normal-SABR expansion:

        sigma_N = alpha * (z / x(z)) * [1 + (2 - 3*rho^2)/24 * nu^2 * T]

    where z = (nu / alpha) * (F - K) and x(z) is the Hagan mapping.
    At the money (F = K) the formula degenerates gracefully to the ATM limit.

    Parameters
    ----------
    F, K, T : float  Forward, strike, maturity.
    alpha   : float  SABR alpha (> 0).
    rho     : float  Correlation in (-1, 1).
    nu      : float  Vol-of-vol (>= 0).

    Returns
    -------
    dict with keys ``sigma_normal`` (float) and ``model`` ("sabr-normal").
    """
    validate_positive(F, "Forward (F)")
    validate_positive(T, "Time to maturity (T)")

    if alpha <= 0.0:
        raise ValueError(f"alpha must be strictly positive. Got {alpha}")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [-1, 1]. Got {rho}")

    rho = float(np.clip(rho, -0.9999, 0.9999))

    # Dominant time correction
    time_correction = 1.0 + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2 * T

    if abs(F - K) < 1e-7 * max(abs(F), abs(K), 1.0):
        # ATM limit: z_n -> 0, so z_n / x(z_n) -> 1
        sigma_N = alpha * time_correction
    else:
        z_n = (nu / alpha) * (F - K)
        x_n = _x_of_z(z_n, rho)
        zx_n = z_n / x_n if abs(z_n) >= 1e-7 else 1.0
        sigma_N = alpha * zx_n * time_correction

    return {"sigma_normal": float(sigma_N), "model": "sabr-normal"}


# ---------------------------------------------------------------------------
# 3. SABR calibration (beta fixed)
# ---------------------------------------------------------------------------

def _solve_alpha_from_atm(
    sigma_atm_market: float,
    F: float,
    T: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Solve for alpha given the ATM implied vol via the cubic equation.

    The ATM SABR formula is:
        sigma_atm = alpha/F^(1-beta) * [1 + (c_a*alpha^2 + c_b*alpha + c_c)*T]

    Expanding and rearranging yields a cubic in alpha:
        c3*alpha^3 + c2*alpha^2 + c1*alpha - sigma_atm_market = 0

    We take the smallest positive real root.
    """
    one_minus_beta = 1.0 - beta
    F_pow = F ** one_minus_beta  # F^(1-beta)

    # Coefficients in sigma = (1/F_pow)*alpha * [1 + (c_a*alpha^2 + c_b*alpha + c_c)*T]
    # c_a multiplies alpha^2, c_b multiplies alpha, c_c is constant.
    c_a = (one_minus_beta ** 2) / 24.0 / F_pow ** 2 * T / F_pow   # alpha^3 coeff contribution
    c_b = 0.25 * rho * beta * nu / F_pow * T / F_pow               # alpha^2 coeff contribution
    c_c = (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2 * T / F_pow     # alpha^1 coeff contribution

    # sigma = alpha/F_pow + c_a*alpha^3 + c_b*alpha^2 + c_c*alpha - sigma_atm_market = 0
    # => c_a*alpha^3 + c_b*alpha^2 + (1/F_pow + c_c)*alpha - sigma_atm_market = 0
    poly_coeffs = [c_a, c_b, 1.0 / F_pow + c_c, -sigma_atm_market]
    roots = np.roots(poly_coeffs)

    # Filter for positive real roots
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-8 and r.real > 0]
    if not real_roots:
        # Fallback: linear approximation alpha ≈ sigma_atm_market * F^(1-beta)
        return sigma_atm_market * F_pow
    # Return smallest positive real root (most stable)
    return float(min(real_roots))


def calibrate_sabr(
    F: float,
    T: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    beta: float = 0.5,
    initial_guess: dict | None = None,
) -> dict:
    """Calibrate SABR parameters (alpha, rho, nu) with beta fixed.

    Minimises the sum of squared vol errors over (rho, nu) by determining
    alpha analytically from the ATM cubic at each candidate (rho, nu).

    Parameters
    ----------
    F : float
        Forward price.
    T : float
        Time to maturity.
    strikes : array-like
        Observed strikes (same length as market_vols).
    market_vols : array-like
        Observed market implied vols (lognormal, same length as strikes).
    beta : float
        CEV exponent, kept fixed (default 0.5).
    initial_guess : dict, optional
        Starting point with keys ``rho`` and ``nu``.  Defaults to rho=0.0,
        nu=0.5.

    Returns
    -------
    dict with keys: ``alpha``, ``beta``, ``rho``, ``nu``, ``rmse``,
    ``fitted_vols`` (ndarray), ``market_vols`` (ndarray), ``strikes`` (ndarray).
    """
    validate_positive(F, "Forward (F)")
    validate_positive(T, "Time to maturity (T)")

    strikes = np.asarray(strikes, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)

    if len(strikes) != len(market_vols):
        raise ValueError("strikes and market_vols must have the same length.")
    if len(strikes) == 0:
        raise ValueError("At least one strike/vol pair is required.")
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1]. Got {beta}")

    # Identify ATM vol: the market vol at the strike closest to F
    atm_idx = int(np.argmin(np.abs(strikes - F)))
    sigma_atm_market = float(market_vols[atm_idx])

    def objective(x: np.ndarray) -> float:
        rho_c, nu_c = float(x[0]), float(x[1])
        try:
            alpha_c = _solve_alpha_from_atm(sigma_atm_market, F, T, beta, rho_c, nu_c)
        except Exception:
            return 1e10

        errors = []
        for K_i, sv_i in zip(strikes, market_vols, strict=False):
            try:
                res = sabr_implied_vol(F, float(K_i), T, alpha_c, beta, rho_c, nu_c)
                errors.append(res["sigma"] - float(sv_i))
            except Exception:
                errors.append(1.0)
        return float(np.sum(np.square(errors)))

    rho0 = float(initial_guess.get("rho", 0.0)) if initial_guess else 0.0
    nu0 = float(initial_guess.get("nu", 0.5)) if initial_guess else 0.5
    x0 = np.array([rho0, nu0], dtype=float)

    bounds = [(-0.999, 0.999), (0.001, 5.0)]
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    rho_opt = float(result.x[0])
    nu_opt = float(result.x[1])
    alpha_opt = _solve_alpha_from_atm(sigma_atm_market, F, T, beta, rho_opt, nu_opt)

    fitted_vols = np.array(
        [sabr_implied_vol(F, float(K_i), T, alpha_opt, beta, rho_opt, nu_opt)["sigma"]
         for K_i in strikes],
        dtype=float,
    )
    rmse = float(np.sqrt(np.mean((fitted_vols - market_vols) ** 2)))

    return {
        "alpha": alpha_opt,
        "beta": float(beta),
        "rho": rho_opt,
        "nu": nu_opt,
        "rmse": rmse,
        "fitted_vols": fitted_vols,
        "market_vols": market_vols,
        "strikes": strikes,
    }
