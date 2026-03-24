import numpy as np
from quant_derivatives.engines.pricing.black_scholes import bs_price_and_greeks
from quant_derivatives.utils.validation import validate_option_params


def _build_vol_proxy(paths: np.ndarray, sigma: float, dt: float, ewma_lambda: float) -> np.ndarray:
    """Builds a simple EWMA implied-vol proxy from realized returns when no vol path is provided."""
    n_steps, n_paths = paths.shape
    vol_proxy = np.full((n_steps, n_paths), float(sigma), dtype=float)
    ewma_var = np.full(n_paths, float(sigma) ** 2, dtype=float)
    for t in range(1, n_steps):
        log_ret = np.log(np.maximum(paths[t], 1e-12) / np.maximum(paths[t - 1], 1e-12))
        inst_var = np.clip((log_ret ** 2) / max(dt, 1e-12), 1e-10, None)
        ewma_var = ewma_lambda * ewma_var + (1.0 - ewma_lambda) * inst_var
        vol_proxy[t] = np.sqrt(np.clip(ewma_var, 1e-10, None))
    return vol_proxy


def simulate_discrete_hedging(
    paths: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
    transaction_cost: float = 0.0,
    implied_vol_path: np.ndarray | None = None,
    ewma_lambda: float = 0.94,
) -> dict:
    """
    Simulates discrete delta hedging and attributes P&L into theta, gamma, vega,
    vanna, volga, transaction costs, and residual/slippage.
    """
    validate_option_params(paths[0, 0], K, T, sigma)
    if transaction_cost < 0:
        raise ValueError("Transaction cost must be non-negative")

    n_steps, n_paths = paths.shape
    dt = T / (n_steps - 1)

    if implied_vol_path is None:
        implied_vol_path = _build_vol_proxy(paths, sigma=sigma, dt=dt, ewma_lambda=ewma_lambda)
    else:
        implied_vol_path = np.asarray(implied_vol_path, dtype=float)
        if implied_vol_path.shape != paths.shape:
            raise ValueError("implied_vol_path must have the same shape as paths")

    total_pnls = np.zeros(n_paths)
    theta_pnls = np.zeros(n_paths)
    gamma_pnls = np.zeros(n_paths)
    vega_pnls = np.zeros(n_paths)
    vanna_pnls = np.zeros(n_paths)
    volga_pnls = np.zeros(n_paths)
    cost_pnls = np.zeros(n_paths)
    residual_pnls = np.zeros(n_paths)
    realized_vols = np.zeros(n_paths)
    avg_proxy_vols = np.zeros(n_paths)

    for i in range(n_paths):
        path = paths[:, i]
        sigma_path = implied_vol_path[:, i]
        cash = bs_price_and_greeks(path[0], K, T, r, max(sigma_path[0], 1e-4), is_call)["price"]
        shares = 0.0

        theta_pnl = 0.0
        gamma_pnl = 0.0
        vega_pnl = 0.0
        vanna_pnl = 0.0
        volga_pnl = 0.0
        total_costs = 0.0

        prev_greeks = None
        prev_S = path[0]
        prev_sigma = sigma_path[0]

        for t in range(n_steps - 1):
            time_to_mat = max(T - t * dt, 1e-10)
            S = path[t]
            sigma_t = max(sigma_path[t], 1e-4)
            greeks = bs_price_and_greeks(S, K, time_to_mat, r, sigma_t, is_call)
            delta = greeks["delta"]

            trade = delta - shares
            cost = abs(trade) * S * transaction_cost
            cash -= trade * S + cost
            shares = delta
            total_costs += cost

            if prev_greeks is not None:
                dS = S - prev_S
                d_sigma = sigma_t - prev_sigma
                d_sigma_points = d_sigma * 100.0

                theta_pnl += prev_greeks["theta"] * 365.0 * dt
                gamma_pnl += 0.5 * prev_greeks["gamma"] * dS ** 2
                vega_pnl += prev_greeks["vega"] * d_sigma_points
                vanna_pnl += prev_greeks.get("vanna", 0.0) * dS * d_sigma_points
                volga_pnl += 0.5 * prev_greeks.get("volga", 0.0) * (d_sigma_points ** 2)

            prev_greeks = greeks
            prev_S = S
            prev_sigma = sigma_t
            cash *= np.exp(r * dt)

        S_T = path[-1]
        payoff = max(S_T - K, 0.0) if is_call else max(K - S_T, 0.0)
        liquidation_cost = abs(shares) * S_T * transaction_cost
        cash += shares * S_T - liquidation_cost
        total_costs += liquidation_cost

        total_pnl = cash - payoff
        explained_pnl = theta_pnl + gamma_pnl + vega_pnl + vanna_pnl + volga_pnl - total_costs
        residual_pnl = total_pnl - explained_pnl

        log_rets = np.diff(np.log(np.maximum(path, 1e-12)))
        realized_vol = np.sqrt(np.sum(log_rets ** 2) / T)

        total_pnls[i] = total_pnl
        theta_pnls[i] = theta_pnl
        gamma_pnls[i] = gamma_pnl
        vega_pnls[i] = vega_pnl
        vanna_pnls[i] = vanna_pnl
        volga_pnls[i] = volga_pnl
        cost_pnls[i] = total_costs
        residual_pnls[i] = residual_pnl
        realized_vols[i] = realized_vol
        avg_proxy_vols[i] = float(np.mean(sigma_path))

    return {
        "total_pnl": total_pnls,
        "theta_pnl": theta_pnls,
        "gamma_pnl": gamma_pnls,
        "vega_pnl": vega_pnls,
        "vanna_pnl": vanna_pnls,
        "volga_pnl": volga_pnls,
        "cost_pnl": cost_pnls,
        "residual_pnl": residual_pnls,
        "realized_vol": realized_vols,
        "avg_proxy_implied_vol": avg_proxy_vols,
    }
