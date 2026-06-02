import numpy as np
from scipy.linalg import solve_banded

from utils.validation import validate_integer_at_least, validate_option_params, validate_probability


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
