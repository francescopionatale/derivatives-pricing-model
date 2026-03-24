import numpy as np
from quant_derivatives.utils.validation import validate_option_params, validate_simulation_params

def binomial_price(S: float, K: float, T: float, r: float, sigma: float, n_steps: int, is_call: bool = True) -> float:
    """
    Computes European option price using a binomial tree (CRR model).
    """
    validate_option_params(S, K, T, sigma)
    validate_simulation_params(n_steps, 1)
    
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    
    prices = np.zeros(n_steps + 1)
    for i in range(n_steps + 1):
        prices[i] = S * (u ** (n_steps - i)) * (d ** i)
        
    if is_call:
        values = np.maximum(0, prices - K)
    else:
        values = np.maximum(0, K - prices)
        
    discount = np.exp(-r * dt)
    for j in range(n_steps - 1, -1, -1):
        for i in range(j + 1):
            values[i] = discount * (p * values[i] + (1 - p) * values[i + 1])
            
    return values[0]

def binomial_price_and_greeks(S: float, K: float, T: float, r: float, sigma: float, n_steps: int, is_call: bool = True) -> dict:
    """
    Computes European option price and Greeks using a binomial tree (CRR model) via numerical differentiation.
    """
    p0 = binomial_price(S, K, T, r, sigma, n_steps, is_call)
    
    # Delta & Gamma (bump S)
    dS = S * 0.01
    p_up = binomial_price(S + dS, K, T, r, sigma, n_steps, is_call)
    p_dn = binomial_price(S - dS, K, T, r, sigma, n_steps, is_call)
    delta = (p_up - p_dn) / (2 * dS)
    gamma = (p_up - 2 * p0 + p_dn) / (dS ** 2)
    
    # Vega (bump sigma)
    dsigma = 0.01
    p_vol_up = binomial_price(S, K, T, r, sigma + dsigma, n_steps, is_call)
    p_vol_dn = binomial_price(S, K, T, r, max(1e-4, sigma - dsigma), n_steps, is_call)
    vega = (p_vol_up - p_vol_dn) / (2 * dsigma)
    
    # Theta (bump T)
    dT = 1.0 / 365.0
    p_T_up = binomial_price(S, K, T + dT, r, sigma, n_steps, is_call)
    p_T_dn = binomial_price(S, K, max(1e-4, T - dT), r, sigma, n_steps, is_call)
    # Theta is dP/dt. Since T is time to maturity, dt = -dT. So Theta = -dP/dT
    theta = -(p_T_up - p_T_dn) / (2 * dT)
    
    # Rho (bump r)
    dr = 0.0001
    p_r_up = binomial_price(S, K, T, r + dr, sigma, n_steps, is_call)
    p_r_dn = binomial_price(S, K, T, r - dr, sigma, n_steps, is_call)
    rho = (p_r_up - p_r_dn) / (2 * dr)
    
    return {
        "price": float(p0),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega) / 100.0,
        "theta": float(theta) / 365.0,
        "rho": float(rho) / 100.0,
        "vanna": 0.0, # Not computed in basic binomial
        "volga": 0.0  # Not computed in basic binomial
    }
