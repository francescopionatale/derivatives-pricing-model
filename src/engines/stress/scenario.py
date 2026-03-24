import numpy as np
from quant_derivatives.utils.validation import validate_option_params, validate_simulation_params


def calculate_var_es(pnls: np.ndarray, confidence_level: float = 0.99) -> dict:
    """
    Calculates Value at Risk (VaR) and Expected Shortfall (ES) from a distribution of P&Ls.
    Returns positive loss magnitudes, not raw P&L percentiles.
    """
    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level must be between 0 and 1")
    if len(pnls) == 0:
        raise ValueError("P&L array must not be empty")

    sorted_pnls = np.sort(np.asarray(pnls, dtype=float))
    n = len(sorted_pnls)
    tail_count = max(1, int(np.ceil((1 - confidence_level) * n)))

    tail = sorted_pnls[:tail_count]
    var_threshold = sorted_pnls[tail_count - 1]

    return {
        "var": float(-var_threshold),
        "es": float(-np.mean(tail)),
        "tail_percentile": float(var_threshold),
    }


def generate_student_t_paths(S0: float, r: float, sigma: float, T: float, n_steps: int, n_paths: int, df: float, seed: int = None) -> np.ndarray:
    """
    Simulates paths using standardized Student-t innovations for heavy tails.
    Requires df > 2 so the standardized innovations have finite variance.
    """
    validate_option_params(S0, 1.0, T, sigma)
    validate_simulation_params(n_steps, n_paths)
    if df <= 2:
        raise ValueError("Degrees of freedom must be greater than 2 for finite variance")

    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S0

    Z = np.random.standard_t(df, size=(n_steps, n_paths))
    Z = Z * np.sqrt((df - 2) / df)

    for t in range(1, n_steps + 1):
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t - 1])

    return paths


def apply_spot_vol_shock(S0: float, sigma: float, spot_shock: float, vol_shock: float) -> tuple:
    """
    Applies a deterministic finite shock to spot and volatility.
    spot_shock: relative shock (e.g., -0.10 for -10%)
    vol_shock: absolute shock (e.g., 0.05 for +5%)
    """
    S_shocked = S0 * (1 + spot_shock)
    sigma_shocked = max(1e-4, sigma + vol_shock)
    return S_shocked, sigma_shocked


def generate_short_convexity_scenario(S0: float, r: float, sigma: float, T: float, n_steps: int, n_paths: int, seed: int = None) -> np.ndarray:
    """
    Generates paths where realized volatility increases significantly when spot drops (negative correlation).
    This stresses portfolios that are short convexity (short gamma/vega).
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S0

    Z = np.random.standard_normal((n_steps, n_paths))

    for t in range(1, n_steps + 1):
        local_vol = sigma * (S0 / paths[t - 1])**1.5
        local_vol = np.clip(local_vol, 0.01, 1.5)

        paths[t] = paths[t - 1] * np.exp((r - 0.5 * local_vol**2) * dt + local_vol * np.sqrt(dt) * Z[t - 1])

    return paths
