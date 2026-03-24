import numpy as np
from scipy.optimize import minimize


def _collect_factor_keys(current_greeks: dict, available_instruments: list[dict], target_constraints: dict, factor_penalties: dict | None) -> list[str]:
    keys = set(current_greeks.keys()) | set(target_constraints.keys())
    if factor_penalties:
        keys |= set(factor_penalties.keys())
    for inst in available_instruments:
        keys |= {k for k in inst.keys() if k != "name"}
    return sorted(keys)


def optimize_portfolio(
    current_greeks: dict,
    available_instruments: list,
    target_constraints: dict,
    factor_penalties: dict | None = None,
    factor_covariance: list[list[float]] | None = None,
    risk_aversion: float = 1.0,
    transaction_costs: list[float] | None = None,
    bounds: list[tuple[float, float]] | None = None,
    gross_limit: float | None = None,
) -> dict:
    """
    Portfolio optimization with exact linear neutrality constraints and a quadratic residual-risk objective.

    The optimizer minimizes: 
        0.5 * x^T Sigma x + 0.5 * risk_aversion * residual^T Lambda residual + transaction_costs * |x|

    where residual = final factor exposures after adding hedge weights.
    """
    if len(available_instruments) == 0:
        raise ValueError("available_instruments must not be empty")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")

    n_instruments = len(available_instruments)
    factor_keys = _collect_factor_keys(current_greeks, available_instruments, target_constraints, factor_penalties)
    key_to_idx = {k: i for i, k in enumerate(factor_keys)}

    B = np.zeros((len(factor_keys), n_instruments), dtype=float)
    for j, inst in enumerate(available_instruments):
        for key, value in inst.items():
            if key != "name":
                B[key_to_idx[key], j] = float(value)

    current = np.array([float(current_greeks.get(k, 0.0)) for k in factor_keys], dtype=float)
    penalties = np.array([float((factor_penalties or {}).get(k, 1.0)) for k in factor_keys], dtype=float)

    if factor_covariance is None:
        Sigma = np.eye(n_instruments)
    else:
        Sigma = np.array(factor_covariance, dtype=float)
        if Sigma.shape != (n_instruments, n_instruments):
            raise ValueError("factor_covariance must be an NxN matrix matching available_instruments")

    if transaction_costs is None:
        tc = np.zeros(n_instruments, dtype=float)
    else:
        tc = np.array(transaction_costs, dtype=float)
        if tc.shape != (n_instruments,):
            raise ValueError("transaction_costs must have one value per instrument")

    if bounds is None:
        opt_bounds = [(-np.inf, np.inf)] * n_instruments
    else:
        if len(bounds) != n_instruments:
            raise ValueError("bounds must have one (lower, upper) pair per instrument")
        opt_bounds = bounds

    constraint_keys = list(target_constraints.keys())
    if constraint_keys:
        Aeq = np.zeros((len(constraint_keys), n_instruments), dtype=float)
        beq = np.zeros(len(constraint_keys), dtype=float)
        for i, key in enumerate(constraint_keys):
            Aeq[i, :] = B[key_to_idx[key], :]
            beq[i] = float(target_constraints[key]) - float(current_greeks.get(key, 0.0))
    else:
        Aeq = np.zeros((0, n_instruments), dtype=float)
        beq = np.zeros(0, dtype=float)

    def residual_exposures(w: np.ndarray) -> np.ndarray:
        return current + B @ w

    def objective(w: np.ndarray) -> float:
        residual = residual_exposures(w)
        risk_term = 0.5 * risk_aversion * float(np.sum(penalties * residual ** 2))
        inventory_term = 0.5 * float(w.T @ Sigma @ w)
        tc_term = float(np.sum(tc * np.sqrt(w**2 + 1e-10)))
        return inventory_term + risk_term + tc_term

    constraints = []
    if constraint_keys:
        constraints.append({"type": "eq", "fun": lambda w: Aeq @ w - beq})
    if gross_limit is not None:
        constraints.append({"type": "ineq", "fun": lambda w: gross_limit - np.sum(np.abs(w))})

    w0 = np.zeros(n_instruments, dtype=float)
    res = minimize(objective, w0, method="SLSQP", bounds=opt_bounds, constraints=constraints)

    if not res.success:
        return {"success": False, "message": res.message, "weights": None}

    weights = np.array(res.x, dtype=float)
    final_exposures = residual_exposures(weights)

    constrained_residuals = {key: float(final_exposures[key_to_idx[key]] - target_constraints[key]) for key in constraint_keys}
    unconstrained_exposures = {key: float(final_exposures[key_to_idx[key]]) for key in factor_keys if key not in constraint_keys}

    return {
        "success": True,
        "objective_value": float(objective(weights)),
        "weights": {inst["name"]: float(weights[j]) for j, inst in enumerate(available_instruments)},
        "final_greeks": {key: float(final_exposures[key_to_idx[key]]) for key in factor_keys},
        "constraint_residuals": constrained_residuals,
        "unconstrained_exposures": unconstrained_exposures,
    }
