import numpy as np

from quant_derivatives.engines.simulation.heston import simulate_heston_paths, check_feller_condition
from quant_derivatives.utils.validation import validate_option_params, validate_simulation_params, validate_heston_params


def heston_vanilla_price_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
    n_steps: int,
    n_paths: int,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    v0: float,
    seed: int | None = None,
    antithetic: bool = False,
) -> dict:
    """Monte Carlo pricing for a vanilla European option under Heston stochastic volatility."""
    validate_option_params(S0, K, T, max(np.sqrt(max(v0, 1e-12)), 1e-6))
    validate_simulation_params(n_steps, n_paths)
    validate_heston_params(kappa, theta, sigma_v, rho, v0)

    paths, variances = simulate_heston_paths(
        S0=S0,
        v0=v0,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        r=r,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )

    terminal_spots = paths[-1]
    payoffs = np.maximum(terminal_spots - K, 0.0) if is_call else np.maximum(K - terminal_spots, 0.0)
    discounted = np.exp(-r * T) * payoffs
    price = float(np.mean(discounted))
    std_err = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
    avg_terminal_var = float(np.mean(np.maximum(variances[-1], 0.0)))
    avg_realized_var = float(np.mean(np.sum(np.maximum(variances[:-1], 0.0), axis=0) * (T / n_steps) / T))

    return {
        "price": price,
        "std_err": std_err,
        "ci_95": [float(price - 1.96 * std_err), float(price + 1.96 * std_err)],
        "feller_condition": bool(check_feller_condition(kappa, theta, sigma_v)),
        "average_terminal_variance": avg_terminal_var,
        "average_realized_variance": avg_realized_var,
        "settings": {
            "n_steps": int(n_steps),
            "n_paths": int(n_paths),
            "seed": seed,
            "antithetic": bool(antithetic),
            "kappa": float(kappa),
            "theta": float(theta),
            "sigma_v": float(sigma_v),
            "rho": float(rho),
            "v0": float(v0),
        },
    }
