def validate_positive(value: float, name: str):
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive. Got {value}")


def validate_non_negative(value: float, name: str):
    if value < 0:
        raise ValueError(f"{name} must be non-negative. Got {value}")


def validate_probability(value: float, name: str):
    if not (0 <= value <= 1):
        raise ValueError(f"{name} must be between 0 and 1. Got {value}")


def validate_integer_at_least(value: int, minimum: int, name: str):
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer. Got {value}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}. Got {value}")


def validate_option_params(S0: float, K: float, T: float, sigma: float, *, allow_zero_maturity: bool = False):
    validate_positive(S0, "Spot price (S0)")
    validate_positive(K, "Strike price (K)")
    if allow_zero_maturity:
        validate_non_negative(T, "Time to maturity (T)")
    else:
        validate_positive(T, "Time to maturity (T)")
    validate_positive(sigma, "Volatility (sigma)")


def validate_simulation_params(n_steps: int, n_paths: int):
    validate_integer_at_least(n_steps, 1, "Number of steps (n_steps)")
    validate_integer_at_least(n_paths, 1, "Number of paths (n_paths)")
    if n_steps < 2:
        raise ValueError("Number of steps (n_steps) must be at least 2 to simulate a time grid")


def validate_heston_params(kappa: float, theta: float, sigma_v: float, rho: float, v0: float):
    validate_positive(kappa, "Mean reversion speed (kappa)")
    validate_positive(theta, "Long-term variance (theta)")
    validate_positive(sigma_v, "Volatility of variance (sigma_v)")
    validate_positive(v0, "Initial variance (v0)")
    if not (-1 <= rho <= 1):
        raise ValueError(f"Correlation (rho) must be between -1 and 1. Got {rho}")
