from scipy.optimize import brentq
from quant_derivatives.engines.pricing.black_scholes import bs_price_and_greeks
from quant_derivatives.utils.validation import validate_option_params

def implied_volatility(target_price: float, S: float, K: float, T: float, r: float, is_call: bool = True) -> float:
    """
    Solves for implied volatility using Brent's method.
    """
    validate_option_params(S, K, T, 0.1) # dummy sigma
    if target_price <= 0:
        return float('nan')
        
    def objective(sigma):
        return bs_price_and_greeks(S, K, T, r, sigma, is_call)["price"] - target_price
        
    try:
        return brentq(objective, 1e-4, 5.0)
    except ValueError:
        return float('nan')
