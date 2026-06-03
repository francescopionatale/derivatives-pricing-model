from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from utils.validation import (
    validate_heston_params,
    validate_integer_at_least,
    validate_option_params,
    validate_probability,
)


def bs_pde_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
    is_american: bool = False,
    N_x: int = 200,
    N_t: int = 100,
    x_width: float = 4.0,
    theta_scheme: float = 0.5,
) -> dict:
    """
    Price a European or American option by solving the Black-Scholes PDE
    in log-spot coordinates via a theta-scheme (Crank-Nicolson at theta=0.5).

    Parameters
    ----------
    S0 : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to maturity (years).
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    is_call : bool
        True for call, False for put.
    is_american : bool
        If True, apply early-exercise constraint at each time step.
    N_x : int
        Number of spatial grid intervals (N_x+1 nodes).
    N_t : int
        Number of time steps.
    x_width : float
        Half-width of the log-spot grid in units of sigma*sqrt(T).
    theta_scheme : float
        Implicitness parameter: 0=explicit, 0.5=Crank-Nicolson, 1=fully implicit.

    Returns
    -------
    dict with keys: price, delta, gamma, grid_S, grid_V, method, is_american, settings.
    """
    # --- Validation ---
    validate_option_params(S0, K, T, sigma)
    validate_integer_at_least(N_x, 2, "N_x")
    validate_integer_at_least(N_t, 1, "N_t")
    validate_probability(theta_scheme, "theta_scheme")

    # --- Grid setup ---
    x0 = np.log(S0)
    half_width = x_width * sigma * np.sqrt(T)
    x_min = x0 - half_width
    x_max = x0 + half_width

    dx = (x_max - x_min) / N_x
    dt = T / N_t

    x = x_min + np.arange(N_x + 1) * dx  # shape (N_x+1,)
    S = np.exp(x)                          # shape (N_x+1,)

    # --- Terminal condition ---
    if is_call:
        V = np.maximum(S - K, 0.0)
    else:
        V = np.maximum(K - S, 0.0)

    # --- Constant PDE coefficients (flat sigma, no j-dependence) ---
    mu = r - 0.5 * sigma**2   # drift in log-space

    # theta-scheme implicit part (multiplied by theta_scheme)
    a = theta_scheme * dt * (sigma**2 / (2 * dx**2) - mu / (2 * dx))
    b = -theta_scheme * dt * (sigma**2 / dx**2 + r)
    c = theta_scheme * dt * (sigma**2 / (2 * dx**2) + mu / (2 * dx))

    # explicit part coefficients (multiplied by (1-theta_scheme))
    th1 = 1.0 - theta_scheme
    a_e = th1 * dt * (sigma**2 / (2 * dx**2) - mu / (2 * dx))
    b_e = -th1 * dt * (sigma**2 / dx**2 + r)
    c_e = th1 * dt * (sigma**2 / (2 * dx**2) + mu / (2 * dx))

    # --- Build LHS banded matrix (constant, built once outside time loop) ---
    # Interior nodes: j = 1 .. N_x-1  -> n_int = N_x-1 unknowns
    n_int = N_x - 1

    # solve_banded format (1,1): row 0 = superdiag, row 1 = diag, row 2 = subdiag
    # LHS:  -a * V[j-1] + (1-b) * V[j] - c * V[j+1]
    ab = np.zeros((3, n_int))
    ab[0, 1:] = -c           # superdiagonal (offset +1), skip first element
    ab[1, :] = 1.0 - b       # main diagonal
    ab[2, :-1] = -a          # subdiagonal (offset -1), skip last element

    # Explicit part diagonal/off-diagonal scalars
    diag_e = 1.0 + b_e
    off_lo_e = a_e    # coefficient for V[j-1]
    off_hi_e = c_e    # coefficient for V[j+1]

    # --- Time stepping (backward from T to 0) ---
    for n in range(N_t):
        # After this step we'll be at T - (n+1)*dt
        tau_after = (n + 1) * dt   # time elapsed from terminal

        # --- Boundary conditions at the new time level ---
        if is_call:
            bc_left = 0.0
            bc_right = np.exp(x_max) - K * np.exp(-r * tau_after)
        else:
            bc_left = K * np.exp(-r * tau_after) - np.exp(x_min)
            bc_right = 0.0

        # --- Assemble RHS from the explicit part of current V ---
        V_int = V[1:N_x]        # interior V at current time, shape (n_int,)
        V_lo = V[0:N_x - 1]    # V[j-1] for j=1..N_x-1
        V_hi = V[2:N_x + 1]    # V[j+1] for j=1..N_x-1

        rhs = off_lo_e * V_lo + diag_e * V_int + off_hi_e * V_hi

        # Incorporate implicit boundary contributions into RHS
        # j=1 node: LHS implicit term has -a*V[0]; bring new BC to RHS
        rhs[0] += a * bc_left
        # j=N_x-1 node: LHS implicit term has -c*V[N_x]; bring new BC to RHS
        rhs[-1] += c * bc_right

        # --- Solve tridiagonal system ---
        V_new_int = solve_banded((1, 1), ab, rhs)

        # --- Reconstruct full V ---
        V_new = np.empty(N_x + 1)
        V_new[0] = bc_left
        V_new[1:N_x] = V_new_int
        V_new[N_x] = bc_right

        # --- American early-exercise constraint ---
        if is_american:
            if is_call:
                payoff = np.maximum(S - K, 0.0)
            else:
                payoff = np.maximum(K - S, 0.0)
            V_new = np.maximum(V_new, payoff)

        V = V_new

    # --- Greeks from the final grid ---
    i0 = int(np.argmin(np.abs(x - x0)))
    # Guard: ensure i0 is not at the boundary
    i0 = max(1, min(i0, N_x - 1))

    dVdx = (V[i0 + 1] - V[i0 - 1]) / (2.0 * dx)
    d2Vdx2 = (V[i0 + 1] - 2.0 * V[i0] + V[i0 - 1]) / (dx**2)

    delta = dVdx / S0
    gamma = (d2Vdx2 - dVdx) / S0**2

    method = "pde-cn" if theta_scheme == 0.5 else "pde-theta"

    return {
        "price": float(V[i0]),
        "delta": float(delta),
        "gamma": float(gamma),
        "grid_S": S,
        "grid_V": V,
        "method": method,
        "is_american": is_american,
        "settings": {
            "N_x": N_x,
            "N_t": N_t,
            "x_width": x_width,
            "theta_scheme": theta_scheme,
        },
    }


# ---------------------------------------------------------------------------
# Heston 2-D PDE: Douglas-Rachford ADI
# ---------------------------------------------------------------------------

def heston_pde_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    v0: float,
    is_call: bool = True,
    N_x: int = 80,
    N_v: int = 40,
    N_t: int = 80,
    x_width: float = 3.0,
    v_max: float | None = None,
    c_v: float = 5.0,
) -> dict:
    """Price a Heston model European option via Douglas-Rachford ADI on a 2-D PDE grid.

    Solves the Heston PDE backward from T to 0 in (x, v) where x = log(S):

        ∂V/∂τ = ½v·∂²V/∂x² + ρσ_v·v·∂²V/∂x∂v + ½σ_v²·v·∂²V/∂v²
              + (r−½v)·∂V/∂x + κ(θ−v)·∂V/∂v − r·V

    Uses the Douglas-Rachford (1956) ADI splitting with θ=1/2 (CN accuracy):
      F₀ = mixed ∂²/∂x∂v (explicit throughout);
      F₁ = x-direction terms (implicit corrector 1);
      F₂ = v-direction terms (implicit corrector 2).

    v-grid: sinh-based non-uniform, concentrating nodes near v = 0.

    Cross-validates against heston_price_cos to within ~1% for typical parameters.

    Returns
    -------
    dict with ``price``, ``method``, ``grid_S``, ``grid_v``, ``grid_V``, ``settings``.
    """
    validate_option_params(S0, K, T, 1e-6)
    validate_heston_params(kappa, theta, sigma_v, rho, v0)
    validate_integer_at_least(N_x, 4, "N_x")
    validate_integer_at_least(N_v, 4, "N_v")
    validate_integer_at_least(N_t, 4, "N_t")

    if v_max is None:
        v_max = max(5.0 * theta, 3.0 * v0, 0.5)

    # ------------------------------------------------------------------ Grids
    x0 = np.log(S0)
    width = x_width * np.sqrt(theta * T)
    x_min = x0 - width
    x_max = x0 + width
    dx = (x_max - x_min) / N_x
    x = x_min + np.arange(N_x + 1) * dx          # (N_x+1,)
    S_grid = np.exp(x)

    # Sinh-transformed v-grid: concentrates nodes near v=0
    xi = np.arange(N_v + 1) / N_v                 # uniform [0,1]
    v = v_max * np.sinh(c_v * xi) / np.sinh(c_v)  # (N_v+1,), v[0]=0
    dv_p = np.diff(v)   # h_plus[j]  = v[j+1]-v[j],  shape (N_v,)
    dv_m = dv_p.copy()
    dv_m[1:] = dv_p[:-1]                          # h_minus[j] = v[j]-v[j-1], j=1..N_v-1

    dt = T / N_t
    theta_adi = 0.5  # CN parameter

    # ---------------------------------------------------------------- Terminal
    def payoff_fn(Si: np.ndarray, tau: float) -> np.ndarray:
        if is_call:
            return np.maximum(Si - K, 0.0)
        return np.maximum(K - Si, 0.0)

    # V[i, j]: i=x-index (0..N_x), j=v-index (0..N_v)
    V = np.zeros((N_x + 1, N_v + 1))
    for j in range(N_v + 1):
        V[:, j] = payoff_fn(S_grid, 0.0)

    # ------------------------------------------------------- BC helpers
    def _apply_bc(Vi: np.ndarray, tau: float) -> np.ndarray:
        """Overwrite boundary rows/cols with Dirichlet values."""
        disc = np.exp(-r * tau)
        # x boundaries
        Vi[0, :] = 0.0 if is_call else np.maximum(K * disc - np.exp(x_min), 0.0)
        Vi[N_x, :] = np.maximum(np.exp(x_max) - K * disc, 0.0) if is_call else 0.0
        # v = 0: diffusion vanishes → option value = BS/intrinsic at σ→0
        for i in range(N_x + 1):
            Vi[i, 0] = max(S_grid[i] - K * disc, 0.0) if is_call else max(K * disc - S_grid[i], 0.0)
        # v = v_max: large vol → deep in/out approximation
        for i in range(N_x + 1):
            Vi[i, N_v] = (S_grid[i] if is_call else 0.0)
        return Vi

    V = _apply_bc(V, 0.0)

    # ------------------------------------------------- Pre-compute FD stencils

    # x-direction stencils (for each j): stored as 1D arrays of length N_x-1
    # F₁: alpha_j*V[i-1,j] + beta_j*V[i,j] + gamma_j*V[i+1,j]
    def _x_stencil(vj: float) -> tuple[float, float, float]:
        a = 0.5 * vj / dx**2 - (r - 0.5 * vj) / (2 * dx)
        b = -vj / dx**2 - r
        c = 0.5 * vj / dx**2 + (r - 0.5 * vj) / (2 * dx)
        return a, b, c

    # v-direction stencils (for each j interior): P_j*V[i,j-1] + Q_j*V[i,j] + R_j*V[i,j+1]
    P_v = np.zeros(N_v + 1)
    Q_v = np.zeros(N_v + 1)
    R_v = np.zeros(N_v + 1)
    for j in range(1, N_v):
        hp = dv_p[j]
        hm = dv_m[j]
        vj = v[j]
        # second deriv FD (non-uniform)
        d2 = sigma_v**2 * vj * 0.5
        P_v[j] = d2 * 2 / (hm * (hp + hm)) - kappa * (theta - vj) / (hp + hm)
        Q_v[j] = -d2 * 2 / (hp * hm)
        R_v[j] = d2 * 2 / (hp * (hp + hm)) + kappa * (theta - vj) / (hp + hm)

    # ----------------------------------------------- ADI time-stepping
    for n in range(N_t):
        tau_new = (n + 1) * dt

        # ---- Step 0: explicit predictor (all operators applied to V_n)
        U0 = V.copy()

        for j in range(1, N_v):
            vj = v[j]
            # x-direction explicit contribution (F₁ * V_n)
            a, b, c = _x_stencil(vj)
            fx = a * V[:-2, j] + b * V[1:-1, j] + c * V[2:, j]
            U0[1:-1, j] += dt * fx

        for j in range(1, N_v):
            # v-direction explicit contribution (F₂ * V_n)
            fv = P_v[j] * V[1:-1, j - 1] + Q_v[j] * V[1:-1, j] + R_v[j] * V[1:-1, j + 1]
            U0[1:-1, j] += dt * fv

        # Mixed term F₀ (ρ*σ_v*v * ∂²V/∂x∂v), central differences
        for j in range(1, N_v):
            vj = v[j]
            hp = dv_p[j]
            hm = dv_m[j]
            coeff = rho * sigma_v * vj / (2 * dx * (hp + hm))
            mix = coeff * (V[2:, j + 1] - V[:-2, j + 1] - V[2:, j - 1] + V[:-2, j - 1])
            U0[1:-1, j] += dt * mix

        # ---- Step 1: correct x-direction
        #  (I - θ*dt*F₁)*V1 = U0 - θ*dt*F₁*V_n
        V1 = U0.copy()
        for j in range(1, N_v):
            vj = v[j]
            a, b, c = _x_stencil(vj)
            n_int = N_x - 1

            # RHS = U0 - θ*dt*(F₁*V_n − F₁*V_n contributions at boundaries)
            rhs = U0[1:-1, j].copy()
            # subtract θ*dt*F₁*V_n (re-apply explicit x-direction and subtract)
            fx_n = a * V[:-2, j] + b * V[1:-1, j] + c * V[2:, j]
            rhs -= theta_adi * dt * fx_n

            # Boundary contributions into RHS
            rhs[0] += theta_adi * dt * a * V1[0, j]   # V1 BC at x=x_min already set
            rhs[-1] += theta_adi * dt * c * V1[N_x, j]

            # Build banded LHS: (I - θ*dt*F₁)
            ab = np.zeros((3, n_int))
            ab[0, 1:] = -theta_adi * dt * c        # superdiag
            ab[1, :] = 1.0 - theta_adi * dt * b    # diag
            ab[2, :-1] = -theta_adi * dt * a        # subdiag

            V1[1:-1, j] = solve_banded((1, 1), ab, rhs)

        # ---- Step 2: correct v-direction
        #  (I - θ*dt*F₂)*V_{n+1} = V1 - θ*dt*F₂*V_n
        V_new = V1.copy()
        for i in range(1, N_x):
            n_int_v = N_v - 1
            rhs = V1[i, 1:-1].copy()

            # subtract θ*dt*F₂*V_n
            fv_n = P_v[1:-1] * V[i, :-2] + Q_v[1:-1] * V[i, 1:-1] + R_v[1:-1] * V[i, 2:]
            rhs -= theta_adi * dt * fv_n

            # Boundary contributions
            rhs[0] += theta_adi * dt * P_v[1] * V_new[i, 0]
            rhs[-1] += theta_adi * dt * R_v[N_v - 1] * V_new[i, N_v]

            ab = np.zeros((3, n_int_v))
            ab[0, 1:] = -theta_adi * dt * R_v[1:N_v - 1]
            ab[1, :] = 1.0 - theta_adi * dt * Q_v[1:N_v]
            ab[2, :-1] = -theta_adi * dt * P_v[2:N_v]

            V_new[i, 1:-1] = solve_banded((1, 1), ab, rhs)

        _apply_bc(V_new, tau_new)
        V = V_new

    # ------------------------------------------------ Interpolate to (S0, v0)
    i0 = int(np.searchsorted(x, x0))
    i0 = max(1, min(i0, N_x - 1))
    j0 = int(np.searchsorted(v, v0))
    j0 = max(1, min(j0, N_v - 1))

    # Bilinear interpolation
    wx = (x0 - x[i0 - 1]) / (x[i0] - x[i0 - 1])
    wv = (v0 - v[j0 - 1]) / (v[j0] - v[j0 - 1])
    price = (
        (1 - wx) * (1 - wv) * V[i0 - 1, j0 - 1]
        + wx * (1 - wv) * V[i0, j0 - 1]
        + (1 - wx) * wv * V[i0 - 1, j0]
        + wx * wv * V[i0, j0]
    )

    return {
        "price": float(max(price, 0.0)),
        "method": "heston-pde-adi",
        "grid_S": S_grid,
        "grid_v": v,
        "grid_V": V,
        "settings": {"N_x": N_x, "N_v": N_v, "N_t": N_t, "v_max": v_max, "c_v": c_v},
    }
