import numpy as np
from scipy.interpolate import griddata


def check_no_arbitrage(
    strikes: np.ndarray,
    maturities: np.ndarray,
    prices: np.ndarray,
    is_call: bool = True,
    S0: float = None,
    r: float = None,
) -> list:
    """
    Performs discrete static no-arbitrage checks.
    Assumes data refers to a single option type (all calls or all puts).
    Optional S0 and r enable simple lower/upper pricing bound checks.
    """
    if not (len(strikes) == len(maturities) == len(prices)):
        raise ValueError("Inputs strikes, maturities, and prices must have the same length")

    issues = []
    unique_maturities = np.unique(maturities)

    for T in unique_maturities:
        idx = maturities == T
        K = np.asarray(strikes[idx], dtype=float)
        P = np.asarray(prices[idx], dtype=float)

        sort_idx = np.argsort(K)
        K = K[sort_idx]
        P = P[sort_idx]

        for i in range(len(K) - 1):
            if is_call and P[i] < P[i + 1]:
                issues.append(f"Call price increasing with strike at T={T}, K1={K[i]}, K2={K[i + 1]}")
            elif not is_call and P[i] > P[i + 1]:
                issues.append(f"Put price decreasing with strike at T={T}, K1={K[i]}, K2={K[i + 1]}")

        for i in range(len(K) - 2):
            w = (K[i + 1] - K[i]) / (K[i + 2] - K[i])
            expected_max = (1 - w) * P[i] + w * P[i + 2]
            if P[i + 1] > expected_max:
                issues.append(f"Convexity violation at T={T}, K1={K[i]}, K2={K[i + 1]}, K3={K[i + 2]}")

        if S0 is not None and r is not None:
            discount = np.exp(-r * T)
            for strike, price in zip(K, P):
                if is_call:
                    lower = max(0.0, S0 - strike * discount)
                    upper = S0
                    kind = 'Call'
                else:
                    lower = max(0.0, strike * discount - S0)
                    upper = strike * discount
                    kind = 'Put'
                if price < lower - 1e-8:
                    issues.append(f"{kind} lower bound violation at T={T}, K={strike}, price={price}, lower_bound={lower}")
                if price > upper + 1e-8:
                    issues.append(f"{kind} upper bound violation at T={T}, K={strike}, price={price}, upper_bound={upper}")

    unique_strikes = np.unique(strikes)
    for K_val in unique_strikes:
        idx = strikes == K_val
        T = np.asarray(maturities[idx], dtype=float)
        P = np.asarray(prices[idx], dtype=float)

        sort_idx = np.argsort(T)
        T = T[sort_idx]
        P = P[sort_idx]

        for i in range(len(T) - 1):
            if P[i] > P[i + 1]:
                issues.append(f"Calendar arbitrage at K={K_val}, T1={T[i]}, T2={T[i + 1]}")

    return issues


def check_put_call_parity(quotes, S0: float, r: float, tolerance: float = 1e-3) -> list:
    """Checks put-call parity where both call and put quotes are available for the same strike and maturity."""
    grouped = {}
    for q in quotes:
        key = (float(q.strike), float(q.maturity))
        grouped.setdefault(key, {})['C' if q.is_call else 'P'] = float(q.mid_price)

    issues = []
    for (strike, maturity), pair in grouped.items():
        if 'C' in pair and 'P' in pair:
            lhs = pair['C'] - pair['P']
            rhs = S0 - strike * np.exp(-r * maturity)
            diff = lhs - rhs
            if abs(diff) > tolerance:
                issues.append(
                    f"Put-call parity violation at K={strike}, T={maturity}, lhs={lhs}, rhs={rhs}, diff={diff}"
                )
    return issues


def calibrate_surface_with_smoothing(
    strikes: np.ndarray,
    maturities: np.ndarray,
    implied_vols: np.ndarray,
    bid_ask_spreads: np.ndarray = None,
    vega: np.ndarray = None,
) -> tuple:
    """
    Builds a smoothed implied-volatility surface using weighted kernel smoothing.
    The weights are liquidity-aware: tighter spreads or higher Vega receive more influence.
    """
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    implied_vols = np.asarray(implied_vols, dtype=float)

    weights = np.ones_like(implied_vols, dtype=float)
    if bid_ask_spreads is not None:
        spreads = np.asarray(bid_ask_spreads, dtype=float)
        weights = 1.0 / np.maximum(spreads, 1e-4)
    elif vega is not None:
        weights = np.maximum(np.asarray(vega, dtype=float), 1e-8)
    weights = weights / np.sum(weights)

    grid_k_vals = np.linspace(np.min(strikes), np.max(strikes), 50)
    grid_t_vals = np.linspace(np.min(maturities), np.max(maturities), 50)
    grid_K, grid_T = np.meshgrid(grid_k_vals, grid_t_vals)

    strike_scale = max(np.ptp(strikes), 1e-8)
    maturity_scale = max(np.ptp(maturities), 1e-8)
    points = np.column_stack(((strikes - np.min(strikes)) / strike_scale, (maturities - np.min(maturities)) / maturity_scale))
    grid_points = np.column_stack(((grid_K.ravel() - np.min(strikes)) / strike_scale, (grid_T.ravel() - np.min(maturities)) / maturity_scale))

    if len(implied_vols) < 3:
        grid_IV = griddata((strikes, maturities), implied_vols, (grid_K, grid_T), method='nearest')
        return grid_K, grid_T, grid_IV

    pairwise = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2))
    positive_pairwise = pairwise[pairwise > 0]
    bandwidth = float(np.median(positive_pairwise)) if positive_pairwise.size else 0.1
    bandwidth = max(bandwidth, 0.05)

    dist2 = np.sum((grid_points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    kernel = np.exp(-0.5 * dist2 / (bandwidth ** 2))
    effective_weights = kernel * weights[None, :]
    weighted_sum = effective_weights @ implied_vols
    normalization = np.sum(effective_weights, axis=1)
    grid_IV = (weighted_sum / np.maximum(normalization, 1e-12)).reshape(grid_K.shape)

    return grid_K, grid_T, grid_IV
