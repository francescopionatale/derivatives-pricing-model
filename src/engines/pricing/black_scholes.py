import numpy as np
from scipy.stats import norm
from quant_derivatives.utils.validation import validate_option_params


def bs_price_and_greeks(S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True) -> dict:
    """
    Computes Black-Scholes price and Greeks.
    Vega is per volatility point (1%).
    Theta is daily (1/365).
    Supports T == 0 by returning the intrinsic value and degenerate Greeks.
    """
    validate_option_params(S, K, T, sigma, allow_zero_maturity=True)

    if T == 0:
        payoff = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        if is_call:
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {
            "price": payoff,
            "delta": delta,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
            "vanna": 0.0,
            "volga": 0.0,
        }

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    phi_d1 = norm.pdf(d1)

    if is_call:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-S * phi_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-S * phi_d1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    gamma = phi_d1 / (S * sigma * np.sqrt(T))
    vega = S * phi_d1 * np.sqrt(T)
    vanna = -phi_d1 * d2 / sigma
    volga = vega * d1 * d2 / sigma

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100.0,
        "theta": theta / 365.0,
        "rho": rho / 100.0,
        "vanna": vanna / 100.0,
        "volga": volga / 10000.0,
    }
