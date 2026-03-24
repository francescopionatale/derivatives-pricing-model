from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from quant_derivatives.engines.pricing.heston_vanilla import heston_vanilla_price_mc
from quant_derivatives.engines.pricing.implied_vol import implied_volatility
from quant_derivatives.models.domain import OptionQuote
from quant_derivatives.utils.validation import validate_positive, validate_non_negative


@dataclass
class HestonCalibrationResult:
    params: dict
    objective_value: float
    rmse_price: float
    rmse_iv: float | None
    n_quotes: int
    success: bool
    message: str
    diagnostics: dict

    def to_dict(self) -> dict:
        return {
            "params": self.params,
            "objective_value": float(self.objective_value),
            "rmse_price": float(self.rmse_price),
            "rmse_iv": None if self.rmse_iv is None else float(self.rmse_iv),
            "n_quotes": int(self.n_quotes),
            "success": bool(self.success),
            "message": self.message,
            "diagnostics": self.diagnostics,
        }


DEFAULT_BOUNDS = {
    "kappa": (0.05, 10.0),
    "theta": (0.005, 0.50),
    "sigma_v": (0.05, 2.0),
    "rho": (-0.95, 0.95),
    "v0": (0.005, 0.50),
}


def _quote_weight(q: OptionQuote, weight_mode: str = "spread") -> float:
    if weight_mode == "uniform":
        return 1.0
    if weight_mode == "spread":
        if q.bid is not None and q.ask is not None and q.ask >= q.bid:
            spread = max(float(q.ask) - float(q.bid), 1e-4)
            return 1.0 / spread
        return 1.0
    raise ValueError(f"Unknown weight_mode={weight_mode}")


def _price_quotes_heston(
    quotes: Iterable[OptionQuote],
    S0: float,
    r: float,
    params: dict,
    n_steps: int,
    n_paths: int,
    seed: int | None,
    antithetic: bool,
) -> tuple[np.ndarray, list[dict]]:
    model_prices = []
    per_quote = []
    for q in quotes:
        res = heston_vanilla_price_mc(
            S0=S0,
            K=float(q.strike),
            T=float(q.maturity),
            r=r,
            is_call=bool(q.is_call),
            n_steps=n_steps,
            n_paths=n_paths,
            kappa=params["kappa"],
            theta=params["theta"],
            sigma_v=params["sigma_v"],
            rho=params["rho"],
            v0=params["v0"],
            seed=seed,
            antithetic=antithetic,
        )
        model_prices.append(res["price"])
        per_quote.append(
            {
                "strike": float(q.strike),
                "maturity": float(q.maturity),
                "is_call": bool(q.is_call),
                "market_price": float(q.mid_price),
                "model_price": float(res["price"]),
                "abs_error": float(abs(res["price"] - q.mid_price)),
                "std_err": float(res["std_err"]),
            }
        )
    return np.array(model_prices, dtype=float), per_quote


def _objective_from_vector(
    x: np.ndarray,
    quotes: list[OptionQuote],
    S0: float,
    r: float,
    weights: np.ndarray,
    n_steps: int,
    n_paths: int,
    seed: int | None,
    antithetic: bool,
) -> float:
    params = {
        "kappa": float(x[0]),
        "theta": float(x[1]),
        "sigma_v": float(x[2]),
        "rho": float(x[3]),
        "v0": float(x[4]),
    }
    model_prices, _ = _price_quotes_heston(quotes, S0, r, params, n_steps, n_paths, seed, antithetic)
    market_prices = np.array([float(q.mid_price) for q in quotes], dtype=float)
    errors = model_prices - market_prices
    weighted_mse = float(np.sum(weights * errors ** 2))

    # Soft penalties encourage reasonable stochastic-vol behaviour during calibration.
    feller_gap = max(params["sigma_v"] ** 2 - 2.0 * params["kappa"] * params["theta"], 0.0)
    long_run_gap = (params["v0"] - params["theta"]) ** 2
    penalty = 10.0 * feller_gap ** 2 + 0.05 * long_run_gap
    return weighted_mse + penalty


def calibrate_heston_to_quotes(
    quotes: list[OptionQuote],
    S0: float,
    r: float,
    n_steps: int = 64,
    n_paths: int = 4000,
    seed: int | None = 42,
    antithetic: bool = True,
    initial_guess: dict | None = None,
    bounds: dict | None = None,
    weight_mode: str = "spread",
    maxiter: int = 30,
) -> HestonCalibrationResult:
    validate_positive(S0, "S0")
    validate_non_negative(r, "r")
    if not quotes:
        raise ValueError("At least one quote is required for Heston calibration")

    bounds = bounds or DEFAULT_BOUNDS
    x0 = np.array(
        [
            float(initial_guess.get("kappa", 2.0) if initial_guess else 2.0),
            float(initial_guess.get("theta", 0.04) if initial_guess else 0.04),
            float(initial_guess.get("sigma_v", 0.30) if initial_guess else 0.30),
            float(initial_guess.get("rho", -0.70) if initial_guess else -0.70),
            float(initial_guess.get("v0", 0.04) if initial_guess else 0.04),
        ],
        dtype=float,
    )
    scipy_bounds = [bounds[k] for k in ("kappa", "theta", "sigma_v", "rho", "v0")]
    weights = np.array([_quote_weight(q, weight_mode) for q in quotes], dtype=float)
    weights = weights / np.sum(weights)

    objective = lambda x: _objective_from_vector(x, quotes, S0, r, weights, n_steps, n_paths, seed, antithetic)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=scipy_bounds,
        options={"maxiter": int(maxiter)},
    )

    params = {
        "kappa": float(result.x[0]),
        "theta": float(result.x[1]),
        "sigma_v": float(result.x[2]),
        "rho": float(result.x[3]),
        "v0": float(result.x[4]),
    }
    model_prices, per_quote = _price_quotes_heston(quotes, S0, r, params, n_steps, n_paths, seed, antithetic)
    market_prices = np.array([float(q.mid_price) for q in quotes], dtype=float)
    rmse_price = float(np.sqrt(np.mean((model_prices - market_prices) ** 2)))

    iv_errors = []
    iv_rows = []
    for q, model_price in zip(quotes, model_prices):
        try:
            market_iv = implied_volatility(float(q.mid_price), S0, float(q.strike), float(q.maturity), r, bool(q.is_call))
            model_iv = implied_volatility(float(model_price), S0, float(q.strike), float(q.maturity), r, bool(q.is_call))
            iv_err = float(model_iv - market_iv)
            iv_errors.append(iv_err)
            iv_rows.append(
                {
                    "strike": float(q.strike),
                    "maturity": float(q.maturity),
                    "is_call": bool(q.is_call),
                    "market_iv": float(market_iv),
                    "model_iv": float(model_iv),
                    "iv_error": iv_err,
                }
            )
        except (RuntimeError, ValueError, OverflowError):
            continue
    rmse_iv = float(np.sqrt(np.mean(np.square(iv_errors)))) if iv_errors else None

    return HestonCalibrationResult(
        params=params,
        objective_value=float(result.fun),
        rmse_price=rmse_price,
        rmse_iv=rmse_iv,
        n_quotes=len(quotes),
        success=bool(result.success),
        message=str(result.message),
        diagnostics={
            "method": "L-BFGS-B",
            "n_steps": int(n_steps),
            "n_paths": int(n_paths),
            "seed": seed,
            "antithetic": bool(antithetic),
            "weight_mode": weight_mode,
            "maxiter": int(maxiter),
            "parameter_bounds": {k: [float(v[0]), float(v[1])] for k, v in bounds.items()},
            "per_quote_pricing": per_quote,
            "per_quote_iv": iv_rows,
            "feller_condition": bool(2.0 * params["kappa"] * params["theta"] > params["sigma_v"] ** 2),
        },
    )
