import numpy as np

from quant_derivatives.engines.simulation.gbm import simulate_gbm_paths
from quant_derivatives.engines.simulation.heston import simulate_heston_paths, check_feller_condition
from quant_derivatives.utils.validation import (
    validate_option_params,
    validate_simulation_params,
    validate_positive,
    validate_heston_params,
)


def _summarize_discounted_payoffs(discounted_payoffs: np.ndarray, n_paths: int) -> dict:
    price = float(np.mean(discounted_payoffs))
    std_err = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths))
    return {
        "price": price,
        "std_err": std_err,
        "ci_95": [float(price - 1.96 * std_err), float(price + 1.96 * std_err)],
    }


def price_barrier_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    is_up: bool,
    is_out: bool,
    is_call: bool,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = False,
) -> dict:
    """Price a barrier option under GBM using Brownian-bridge barrier monitoring correction."""
    validate_option_params(S0, K, T, sigma)
    validate_simulation_params(n_steps, n_paths)
    validate_positive(barrier, "barrier")

    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed, antithetic)
    dt = T / n_steps
    hit_barrier = np.zeros(n_paths, dtype=bool)

    for t in range(n_steps):
        S_t = paths[t]
        S_next = paths[t + 1]
        prob_hit = np.zeros(n_paths)

        if is_up:
            mask = (S_t < barrier) & (S_next < barrier)
            prob_hit[mask] = np.exp(
                -2.0 * np.log(barrier / S_t[mask]) * np.log(barrier / S_next[mask]) / (sigma ** 2 * dt)
            )
            prob_hit[S_next >= barrier] = 1.0
        else:
            mask = (S_t > barrier) & (S_next > barrier)
            prob_hit[mask] = np.exp(
                -2.0 * np.log(barrier / S_t[mask]) * np.log(barrier / S_next[mask]) / (sigma ** 2 * dt)
            )
            prob_hit[S_next <= barrier] = 1.0

        random_draws = np.random.uniform(0.0, 1.0, n_paths)
        hit_barrier |= random_draws < prob_hit

    terminal_spots = paths[-1]
    payoffs = np.maximum(terminal_spots - K, 0.0) if is_call else np.maximum(K - terminal_spots, 0.0)
    if is_out:
        payoffs[hit_barrier] = 0.0
    else:
        payoffs[~hit_barrier] = 0.0

    discounted_payoffs = np.exp(-r * T) * payoffs
    result = _summarize_discounted_payoffs(discounted_payoffs, n_paths)
    result.update(
        {
            "model": "gbm",
            "barrier": float(barrier),
            "direction": "up" if is_up else "down",
            "style": "out" if is_out else "in",
            "brownian_bridge_correction": True,
            "barrier_hit_ratio": float(np.mean(hit_barrier)),
            "settings": {
                "n_steps": int(n_steps),
                "n_paths": int(n_paths),
                "seed": seed,
                "antithetic": bool(antithetic),
            },
        }
    )
    return result



def price_barrier_heston_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    barrier: float,
    is_up: bool,
    is_out: bool,
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
    """Price a barrier option under Heston using discrete path monitoring."""
    validate_option_params(S0, K, T, max(np.sqrt(max(v0, 1e-12)), 1e-6))
    validate_simulation_params(n_steps, n_paths)
    validate_positive(barrier, "barrier")
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

    hit_barrier = np.any(paths >= barrier, axis=0) if is_up else np.any(paths <= barrier, axis=0)
    terminal_spots = paths[-1]
    payoffs = np.maximum(terminal_spots - K, 0.0) if is_call else np.maximum(K - terminal_spots, 0.0)
    if is_out:
        payoffs[hit_barrier] = 0.0
    else:
        payoffs[~hit_barrier] = 0.0

    discounted_payoffs = np.exp(-r * T) * payoffs
    result = _summarize_discounted_payoffs(discounted_payoffs, n_paths)
    result.update(
        {
            "model": "heston",
            "barrier": float(barrier),
            "direction": "up" if is_up else "down",
            "style": "out" if is_out else "in",
            "brownian_bridge_correction": False,
            "monitoring": "discrete",
            "barrier_hit_ratio": float(np.mean(hit_barrier)),
            "feller_condition": bool(check_feller_condition(kappa, theta, sigma_v)),
            "average_terminal_variance": float(np.mean(np.maximum(variances[-1], 0.0))),
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
    )
    return result



def price_lookback_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_floating: bool,
    is_call: bool,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = False,
) -> dict:
    """Price a lookback option under GBM using Brownian-bridge extrema correction."""
    validate_option_params(S0, K, T, sigma)
    validate_simulation_params(n_steps, n_paths)

    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed, antithetic)
    dt = T / n_steps

    s_min = np.full(n_paths, S0, dtype=float)
    s_max = np.full(n_paths, S0, dtype=float)

    for t in range(n_steps):
        s_t = np.maximum(paths[t], 1e-12)
        s_next = np.maximum(paths[t + 1], 1e-12)

        u = np.random.uniform(0.0, 1.0, n_paths)
        x_t = np.log(s_t)
        x_next = np.log(s_next)
        diff = x_next - x_t
        bridge_term = np.sqrt(np.maximum(diff ** 2 - 2.0 * sigma ** 2 * dt * np.log(u), 0.0))

        s_max_interval = np.exp(0.5 * (x_t + x_next + bridge_term))
        s_min_interval = np.exp(0.5 * (x_t + x_next - bridge_term))

        s_min = np.minimum(s_min, s_min_interval)
        s_max = np.maximum(s_max, s_max_interval)

    terminal_spots = paths[-1]
    if is_floating:
        payoffs = np.maximum(terminal_spots - s_min, 0.0) if is_call else np.maximum(s_max - terminal_spots, 0.0)
    else:
        payoffs = np.maximum(s_max - K, 0.0) if is_call else np.maximum(K - s_min, 0.0)

    discounted_payoffs = np.exp(-r * T) * payoffs
    result = _summarize_discounted_payoffs(discounted_payoffs, n_paths)
    result.update(
        {
            "model": "gbm",
            "lookback_style": "floating" if is_floating else "fixed",
            "brownian_bridge_correction": True,
            "average_path_min": float(np.mean(s_min)),
            "average_path_max": float(np.mean(s_max)),
            "settings": {
                "n_steps": int(n_steps),
                "n_paths": int(n_paths),
                "seed": seed,
                "antithetic": bool(antithetic),
            },
        }
    )
    return result



def price_lookback_heston_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    is_floating: bool,
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
    """Price a lookback option under Heston using discrete path extrema."""
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

    s_min = np.min(paths, axis=0)
    s_max = np.max(paths, axis=0)
    terminal_spots = paths[-1]

    if is_floating:
        payoffs = np.maximum(terminal_spots - s_min, 0.0) if is_call else np.maximum(s_max - terminal_spots, 0.0)
    else:
        payoffs = np.maximum(s_max - K, 0.0) if is_call else np.maximum(K - s_min, 0.0)

    discounted_payoffs = np.exp(-r * T) * payoffs
    result = _summarize_discounted_payoffs(discounted_payoffs, n_paths)
    result.update(
        {
            "model": "heston",
            "lookback_style": "floating" if is_floating else "fixed",
            "brownian_bridge_correction": False,
            "monitoring": "discrete",
            "average_path_min": float(np.mean(s_min)),
            "average_path_max": float(np.mean(s_max)),
            "feller_condition": bool(check_feller_condition(kappa, theta, sigma_v)),
            "average_terminal_variance": float(np.mean(np.maximum(variances[-1], 0.0))),
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
    )
    return result
