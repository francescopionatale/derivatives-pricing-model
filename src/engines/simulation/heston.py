import numpy as np
from numpy.random import SeedSequence

from utils.validation import (
    validate_heston_params,
    validate_option_params,
    validate_positive,
    validate_simulation_params,
)

SeedLike = int | SeedSequence | None


def check_feller_condition(kappa: float, theta: float, sigma_v: float) -> bool:
    """
    Checks if the Feller condition (2 * kappa * theta > sigma_v^2) is satisfied.
    If satisfied, the variance process is strictly positive.
    """
    validate_positive(kappa, "kappa")
    validate_positive(theta, "theta")
    validate_positive(sigma_v, "sigma_v")
    return 2 * kappa * theta > sigma_v**2


def simulate_heston_paths_euler(
    S0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    r: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: SeedLike = None,
    antithetic: bool = False,
) -> tuple:
    """
    Simulates Heston model paths using Euler-Maruyama with full truncation for the variance process.
    Returns (S_paths, V_paths) of shape (n_steps + 1, n_paths).

    Randomness uses numpy's modern Generator (np.random.default_rng) for reproducible,
    state-isolated draws.
    """
    validate_option_params(S0, 1.0, T, 0.1)
    validate_heston_params(kappa, theta, sigma_v, rho, v0)
    validate_simulation_params(n_steps, n_paths)

    rng = np.random.default_rng(seed)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    S = np.zeros((n_steps + 1, n_paths))
    V = np.zeros((n_steps + 1, n_paths))

    S[0] = S0
    V[0] = v0

    if antithetic:
        n_half = n_paths // 2
        Z1 = rng.standard_normal((n_steps, n_half))
        Z2 = rng.standard_normal((n_steps, n_half))
        Z1 = np.concatenate((Z1, -Z1), axis=1)
        Z2 = np.concatenate((Z2, -Z2), axis=1)
    else:
        Z1 = rng.standard_normal((n_steps, n_paths))
        Z2 = rng.standard_normal((n_steps, n_paths))

    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

    for t in range(1, n_steps + 1):
        v_prev = np.maximum(V[t - 1], 0)  # Full truncation

        V[t] = v_prev + kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev) * sqrt_dt * W2[t - 1]
        S[t] = S[t - 1] * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev) * sqrt_dt * W1[t - 1])

    return S, V


def simulate_heston_paths_qe(
    S0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    r: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: SeedLike = None,
    antithetic: bool = False,
    psi_c: float = 1.5,
    gamma1: float = 0.5,
) -> tuple:
    """
    Simulates Heston model paths using the Andersen (2008) Quadratic Exponential (QE) scheme.

    The QE scheme is exact for the first two conditional moments of the variance process,
    allowing accurate prices with far fewer time steps than Euler-Maruyama.

    Parameters
    ----------
    psi_c : float
        Threshold that controls the switch between the quadratic (psi <= psi_c)
        and exponential (psi > psi_c) branches. Andersen recommends 1.0-2.0.
    gamma1 : float
        Martingale-correction weighting parameter (gamma2 = 1 - gamma1).
        Andersen recommends 0.5 for the centred discretisation.

    Returns
    -------
    (S_paths, V_paths) each of shape (n_steps + 1, n_paths).
    """
    validate_option_params(S0, 1.0, T, 0.1)
    validate_heston_params(kappa, theta, sigma_v, rho, v0)
    validate_simulation_params(n_steps, n_paths)

    rng = np.random.default_rng(seed)

    dt = T / n_steps
    gamma2 = 1.0 - gamma1

    # Martingale-correction constants (fixed across steps)
    K0 = -(rho * kappa * theta / sigma_v) * dt
    K1 = gamma1 * dt * (kappa * rho / sigma_v - 0.5) - rho / sigma_v
    K2 = gamma2 * dt * (kappa * rho / sigma_v - 0.5) + rho / sigma_v
    K3 = gamma1 * dt * (1.0 - rho**2)
    K4 = gamma2 * dt * (1.0 - rho**2)

    # Pre-allocate path arrays
    S = np.zeros((n_steps + 1, n_paths))
    V = np.zeros((n_steps + 1, n_paths))
    S[0] = S0
    V[0] = v0

    # Draw all random variates upfront for efficiency
    if antithetic:
        n_half = n_paths // 2
        Zv_half = rng.standard_normal((n_steps, n_half))
        Zs_half = rng.standard_normal((n_steps, n_half))
        U_half = rng.uniform(0.0, 1.0, (n_steps, n_half))

        Zv = np.concatenate((Zv_half, -Zv_half), axis=1)
        Zs = np.concatenate((Zs_half, -Zs_half), axis=1)
        # For the exponential branch: antithetic of U is 1-U
        U = np.concatenate((U_half, 1.0 - U_half), axis=1)
    else:
        Zv = rng.standard_normal((n_steps, n_paths))
        Zs = rng.standard_normal((n_steps, n_paths))
        U = rng.uniform(0.0, 1.0, (n_steps, n_paths))

    eps = 1e-14  # guard against V=0 or degenerate moments

    exp_kdt = np.exp(-kappa * dt)
    one_minus_exp = 1.0 - exp_kdt

    for t in range(n_steps):
        v_cur = V[t]  # shape (n_paths,)

        # Conditional moments of V_{t+dt} | V_t
        m = theta + (v_cur - theta) * exp_kdt
        s2 = (
            v_cur * sigma_v**2 * exp_kdt * one_minus_exp / kappa
            + theta * sigma_v**2 * one_minus_exp**2 / (2.0 * kappa)
        )
        m2 = m**2
        psi = s2 / np.maximum(m2, eps)

        # ------------------------------------------------------------------
        # Quadratic branch  (psi <= psi_c)
        # ------------------------------------------------------------------
        b2 = 2.0 / psi - 1.0 + np.sqrt(2.0 / psi) * np.sqrt(np.maximum(2.0 / psi - 1.0, 0.0))
        a_quad = m / (1.0 + b2)
        b_quad = np.sqrt(np.maximum(b2, 0.0))
        v_quad = a_quad * (b_quad + Zv[t]) ** 2

        # ------------------------------------------------------------------
        # Exponential branch  (psi > psi_c)
        # ------------------------------------------------------------------
        p_exp = (psi - 1.0) / (psi + 1.0)
        beta_exp = 2.0 / (m * (psi + 1.0))
        u_t = U[t]
        safe_denom = np.maximum(1.0 - u_t, eps)
        v_exp = np.where(
            u_t <= p_exp,
            0.0,
            np.log(np.maximum((1.0 - p_exp) / safe_denom, eps)) / beta_exp,
        )

        # Select branch per path
        use_quad = psi <= psi_c
        v_next = np.where(use_quad, v_quad, v_exp)
        V[t + 1] = np.maximum(v_next, 0.0)  # numerical safety clamp

        # Log-spot update with martingale correction
        # K0 accounts for the rho-kappa-theta drift adjustment; r*dt is added explicitly.
        variance_term = np.maximum(K3 * v_cur + K4 * V[t + 1], 0.0)
        log_S_next = (
            np.log(S[t]) + r * dt + K0 + K1 * v_cur + K2 * V[t + 1] + np.sqrt(variance_term) * Zs[t]
        )
        S[t + 1] = np.exp(log_S_next)

    return S, V


def simulate_heston_paths(
    S0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    r: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: SeedLike = None,
    antithetic: bool = False,
    scheme: str = "qe",
) -> tuple:
    """
    Dispatcher for Heston path simulation.

    Parameters
    ----------
    scheme : {"qe", "euler"}
        "qe"    - Andersen (2008) Quadratic Exponential scheme (default, more accurate).
        "euler" - Euler-Maruyama with full truncation.

    Returns
    -------
    (S_paths, V_paths) of shape (n_steps + 1, n_paths).
    """
    if scheme == "euler":
        return simulate_heston_paths_euler(
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
    if scheme == "qe":
        return simulate_heston_paths_qe(
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
    raise ValueError(f"Unknown scheme '{scheme}'. Choose 'qe' or 'euler'.")
