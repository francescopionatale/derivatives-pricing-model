import numpy as np
from quant_derivatives.utils.validation import validate_option_params, validate_simulation_params, validate_heston_params, validate_positive

def check_feller_condition(kappa: float, theta: float, sigma_v: float) -> bool:
    """
    Checks if the Feller condition (2 * kappa * theta > sigma_v^2) is satisfied.
    If satisfied, the variance process is strictly positive.
    """
    validate_positive(kappa, "kappa")
    validate_positive(theta, "theta")
    validate_positive(sigma_v, "sigma_v")
    return 2 * kappa * theta > sigma_v ** 2

def simulate_heston_paths(S0: float, v0: float, kappa: float, theta: float, sigma_v: float, rho: float, r: float, T: float, n_steps: int, n_paths: int, seed: int = None, antithetic: bool = False) -> tuple:
    """
    Simulates Heston model paths using Euler-Maruyama with full truncation for the variance process.
    Returns (S_paths, V_paths) of shape (n_steps + 1, n_paths).
    """
    validate_option_params(S0, 1.0, T, 0.1)
    validate_heston_params(kappa, theta, sigma_v, rho, v0)
    validate_simulation_params(n_steps, n_paths)
    
    if seed is not None:
        np.random.seed(seed)
        
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    S = np.zeros((n_steps + 1, n_paths))
    V = np.zeros((n_steps + 1, n_paths))
    
    S[0] = S0
    V[0] = v0
    
    if antithetic:
        n_half = n_paths // 2
        Z1 = np.random.standard_normal((n_steps, n_half))
        Z2 = np.random.standard_normal((n_steps, n_half))
        Z1 = np.concatenate((Z1, -Z1), axis=1)
        Z2 = np.concatenate((Z2, -Z2), axis=1)
    else:
        Z1 = np.random.standard_normal((n_steps, n_paths))
        Z2 = np.random.standard_normal((n_steps, n_paths))
        
    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    for t in range(1, n_steps + 1):
        v_prev = np.maximum(V[t-1], 0) # Full truncation
        
        V[t] = v_prev + kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev) * sqrt_dt * W2[t-1]
        S[t] = S[t-1] * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev) * sqrt_dt * W1[t-1])
        
    return S, V
