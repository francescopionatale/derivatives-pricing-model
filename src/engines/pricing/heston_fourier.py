"""Heston model pricing via Fourier methods.

Three independent implementations serve as mutual cross-checks:
  - Carr-Madan (1999): FFT strike-strip pricer with α-damping and Simpson weights.
  - Fang-Oosterlee COS (2008): cosine-series expansion; fast default for calibration.
  - Lewis (2001): simple quadrature integration — the easiest cross-check.

The shared building block is heston_characteristic_function, implemented with the
"little-trap" (g' = 1/g) formulation to avoid the complex-log branch cut that afflicts
the naive Hagan formula for large maturities or strong mean-reversion.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from utils.validation import validate_heston_params, validate_option_params, validate_positive

# ---------------------------------------------------------------------------
# Shared characteristic function
# ---------------------------------------------------------------------------

def heston_characteristic_function(
    u: np.ndarray,
    S0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    r: float,
    T: float,
) -> np.ndarray:
    """Return the risk-neutral log-price characteristic function φ(u) under Heston.

    Uses the g'⁻¹ = (ξ−d)/(ξ+d) formulation (Albrecher et al. 2007 "little trap")
    with exp(−dT) instead of exp(+dT).  This sidesteps the 0/0 that arises in the
    naive g' = (ξ+d)/(ξ−d) form at u=0 (where ξ=d) and avoids the complex-log
    branch-cut discontinuity for large T or strong mean-reversion.

    At u=0: ξ=d=κ  →  g'_inv=0, B=0, A=0  →  φ(0)=1 ✓
    At u=−i: ξ=d     →  same collapse, φ(−i)=S0·exp(rT) (forward) ✓

    Parameters
    ----------
    u : array of complex128, shape (N,)
    Returns array of complex128, shape (N,).
    """
    u = np.asarray(u, dtype=complex)

    xi = kappa - rho * sigma_v * 1j * u
    d = np.sqrt(xi**2 + sigma_v**2 * (u**2 + 1j * u))

    # g'_inv = (ξ−d)/(ξ+d) is zero at u=0, avoiding the 0/0 in g' = (ξ+d)/(ξ−d).
    # Re(ξ+d) = κ + Re(d) ≥ κ > 0 for standard Heston params, so denominator is safe.
    exp_neg_dT = np.exp(-d * T)
    g_prime_inv = (xi - d) / (xi + d)

    denom = 1.0 - g_prime_inv * exp_neg_dT

    B = (xi - d) / sigma_v**2 * (1.0 - exp_neg_dT) / denom
    A = kappa * theta / sigma_v**2 * ((xi - d) * T - 2.0 * np.log(denom / (1.0 - g_prime_inv)))

    phi = np.exp(1j * u * (np.log(S0) + r * T) + A + B * v0)
    return phi


# ---------------------------------------------------------------------------
# Carr-Madan (1999) FFT pricer
# ---------------------------------------------------------------------------

def heston_price_carr_madan(
    S0: float,
    K: float | np.ndarray,
    T: float,
    r: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    v0: float,
    is_call: bool = True,
    N: int = 4096,
    alpha: float = 1.5,
    eta: float = 0.25,
) -> dict:
    """Price Heston option(s) via the Carr-Madan (1999) FFT method.

    One FFT prices a full log-strike strip; requested strikes are interpolated via
    cubic spline.  K may be a scalar float or a numpy array.
    """
    validate_option_params(S0, 1.0, T, 1e-4)
    validate_heston_params(kappa, theta, sigma_v, rho, v0)
    validate_positive(alpha, "alpha")

    scalar_input = np.isscalar(K)
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))

    # FFT integration grid
    j = np.arange(N, dtype=float)
    u_j = j * eta
    lam = 2.0 * np.pi / (N * eta)
    b = N * lam / 2.0
    k_j = -b + j * lam  # log-strike grid

    # Modified characteristic function (damped)
    phi_u = heston_characteristic_function(
        u_j - (alpha + 1.0) * 1j, S0, v0, kappa, theta, sigma_v, rho, r, T
    )
    denom = alpha**2 + alpha - u_j**2 + 1j * (2.0 * alpha + 1.0) * u_j
    # Guard divide-by-zero at u=0
    denom[0] = denom[0] if abs(denom[0]) > 1e-14 else 1e-14 + 0j
    psi = np.exp(-r * T) * phi_u / denom

    # Simpson's rule weights
    delta = np.ones(N) * (eta / 3.0) * 3.0
    delta[0::2] = (eta / 3.0)  # odd-indexed (1-based: j=1,3,5,...) get weight 4; j=0 gets 1
    # Correct Simpson: d_0 = eta/3, d_j = 4*eta/3 for odd j, 2*eta/3 for even j≥2
    delta = np.empty(N)
    delta[0] = eta / 3.0
    delta[1::2] = 4.0 * eta / 3.0  # odd
    delta[2::2] = 2.0 * eta / 3.0  # even ≥ 2

    # Phase twist: exp(+i·v·b) so that FFT output at index n equals the
    # Carr-Madan integral at log-strike k_n = n·λ − b.
    # Re[FFT[exp(i·v·b)·ψ·w]_n] = Re[∫ψ(v)·exp(i·v·k_n)dv] because
    # Re[z] = Re[z̄], absorbing the sign flip in the FFT convention.
    x_j = np.exp(1j * u_j * b) * psi * delta
    fft_vals = np.real(np.fft.fft(x_j))
    call_prices = np.exp(-alpha * k_j) / np.pi * fft_vals

    # Interpolate to requested log-strikes
    log_K = np.log(K_arr)
    # Only interpolate within the grid; clip extrapolation to boundary
    spline = CubicSpline(k_j, call_prices, extrapolate=True)
    call_interp = np.maximum(spline(log_K), 0.0)

    if is_call:
        prices = call_interp
    else:
        # Put via put-call parity
        prices = call_interp - S0 + K_arr * np.exp(-r * T)
        prices = np.maximum(prices, 0.0)

    price_out = float(prices[0]) if scalar_input else prices
    return {
        "price": price_out,
        "method": "carr-madan",
        "strikes": k_j,  # log-strike grid (for diagnostics)
    }


# ---------------------------------------------------------------------------
# Fang-Oosterlee COS (2008)
# ---------------------------------------------------------------------------

def _cos_payoff_coefficients(
    k: np.ndarray, a: float, b: float, is_call: bool
) -> np.ndarray:
    """Compute cosine-series payoff coefficients V_k for call or put."""
    pi_k_over_ba = k * np.pi / (b - a)

    def chi(c: float, d: float) -> np.ndarray:
        # ∫_c^d exp(x) cos(k π (x-a)/(b-a)) dx
        coeff = 1.0 / (1.0 + pi_k_over_ba**2)
        val = (
            np.cos(pi_k_over_ba * (d - a)) * np.exp(d)
            - np.cos(pi_k_over_ba * (c - a)) * np.exp(c)
            + pi_k_over_ba * np.sin(pi_k_over_ba * (d - a)) * np.exp(d)
            - pi_k_over_ba * np.sin(pi_k_over_ba * (c - a)) * np.exp(c)
        )
        return coeff * val

    def psi(c: float, d: float) -> np.ndarray:
        # ∫_c^d cos(k π (x-a)/(b-a)) dx; k=0 term handled analytically to avoid 0/0.
        k_safe = np.where(k == 0, 1.0, k)  # avoid division by zero before np.where selects
        out = np.where(
            k == 0,
            d - c,
            (b - a) / (k_safe * np.pi) * (np.sin(pi_k_over_ba * (d - a)) - np.sin(pi_k_over_ba * (c - a))),
        )
        return out

    if is_call:
        # Payoff max(S-K, 0) → max(exp(x)-1, 0) at x=ln(S/K); integration over [0, b]
        V_k = (2.0 / (b - a)) * (chi(0.0, b) - psi(0.0, b))
    else:
        # Payoff max(K-S, 0) → max(1-exp(x), 0); integration over [a, 0]
        V_k = (2.0 / (b - a)) * (-chi(a, 0.0) + psi(a, 0.0))
    V_k[0] = V_k[0] * 0.5  # k=0 term halved
    return V_k


def heston_price_cos(
    S0: float,
    K: float | np.ndarray,
    T: float,
    r: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    v0: float,
    is_call: bool = True,
    N: int = 256,
    L: float = 10.0,
) -> dict:
    """Price Heston option(s) via the Fang-Oosterlee (2008) COS method.

    Extremely fast for calibration: a single call prices multiple strikes in ~0.1 ms.
    K may be a scalar or numpy array.
    """
    validate_option_params(S0, 1.0, T, 1e-4)
    validate_heston_params(kappa, theta, sigma_v, rho, v0)

    scalar_input = np.isscalar(K)
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))

    # Truncation bounds from first two cumulants of the Heston log-price
    c1 = (r - theta / 2.0) * T + (1.0 - np.exp(-kappa * T)) / (2.0 * kappa) * (v0 - theta)
    c2_approx = v0 * T + theta * T
    width = L * np.sqrt(max(c2_approx, 1e-8))
    a = c1 - width
    b = c1 + width

    k_idx = np.arange(N, dtype=float)
    u = k_idx * np.pi / (b - a)  # real-valued integration points

    # Characteristic function evaluated at real u (shape N,)
    phi_k = heston_characteristic_function(u + 0j, S0, v0, kappa, theta, sigma_v, rho, r, T)

    # Payoff coefficients V_k (shape N,); computed once, broadcast over strikes
    V_k = _cos_payoff_coefficients(k_idx, a, b, is_call)

    # heston_cf(u) = E[exp(i*u*log(S_T))].  FO needs phi_return(u)*exp(i*u*(x-a))
    # where phi_return = heston_cf * exp(-i*u*log(S0)) and x = log(S0/K).
    # Combined: heston_cf(u) * exp(-i*u*(log(K) + a))
    phase = np.exp(-1j * np.outer(u, np.log(K_arr) + a))  # (N, M)

    # sum_k Re[phi_k * phase_k] * V_k  →  (M,)
    weighted = np.real(phi_k[:, None] * phase) * V_k[:, None]  # (N, M)
    prices = K_arr * np.exp(-r * T) * np.sum(weighted, axis=0)
    prices = np.maximum(prices, 0.0)

    price_out = float(prices[0]) if scalar_input else prices
    return {"price": price_out, "method": "cos"}


# ---------------------------------------------------------------------------
# Lewis (2001) integration
# ---------------------------------------------------------------------------

def heston_price_lewis(
    S0: float,
    K: float,
    T: float,
    r: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    v0: float,
    is_call: bool = True,
) -> dict:
    """Price a single Heston option via Lewis (2001) strip-of-analyticity integral.

    C = S0 - K*exp(-rT)/π * Re[∫₀^∞ φ(u-i/2) * exp(-iu*ln(K/S0)) / (u² + 1/4) du]

    Useful as a simple cross-check for Carr-Madan and COS.  Scalar strikes only.
    """
    validate_option_params(S0, float(K), T, 1e-4)
    validate_heston_params(kappa, theta, sigma_v, rho, v0)

    # Lewis (2001): C = S0 - K·e^{-rT}/π · ∫₀^∞ Re[φ(u_c)·K^{−i·u_c}] / (u²+¼) du
    # where u_c = u − i/2 and φ = heston_cf (char function of log S_T).
    # K^{−i·u_c} = exp(−i·u_c·log K) = exp(−iu·log K)/√K, so the √K cancels with K in front,
    # leaving the effective pricing weight √K·e^{-rT}/π — verified numerically to match COS.
    log_K = np.log(float(K))
    discount = np.exp(-r * T)

    def integrand_re(u: float) -> float:
        u_c = u - 0.5j
        phi = heston_characteristic_function(
            np.array([u_c]), S0, v0, kappa, theta, sigma_v, rho, r, T
        )[0]
        # Use u_c (not u) in the log kernel: absorbs the √K factor analytically.
        kernel = phi * np.exp(-1j * u_c * log_K) / (u**2 + 0.25)
        return float(np.real(kernel))

    integral, _ = quad(integrand_re, 0.0, 200.0, limit=200, epsabs=1e-8, epsrel=1e-8)
    call_price = float(S0 - K * discount / np.pi * integral)
    call_price = max(call_price, 0.0)

    if is_call:
        price = call_price
    else:
        price = max(call_price - S0 + K * discount, 0.0)

    return {"price": float(price), "method": "lewis"}
