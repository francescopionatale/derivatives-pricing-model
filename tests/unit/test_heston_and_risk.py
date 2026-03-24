import numpy as np

from quant_derivatives.engines.pricing.heston_vanilla import heston_vanilla_price_mc
from quant_derivatives.engines.risk.optimization import optimize_portfolio


def test_heston_price_is_positive_and_reports_feller():
    res = heston_vanilla_price_mc(
        S0=100,
        K=100,
        T=1.0,
        r=0.03,
        is_call=True,
        n_steps=50,
        n_paths=2000,
        kappa=2.0,
        theta=0.04,
        sigma_v=0.3,
        rho=-0.7,
        v0=0.04,
        seed=42,
    )
    assert res["price"] > 0.0
    assert len(res["ci_95"]) == 2
    assert isinstance(res["feller_condition"], bool)


def test_risk_optimizer_hits_constraints():
    result = optimize_portfolio(
        current_greeks={"delta": 10.0, "gamma": -2.0, "vega": 5.0},
        available_instruments=[
            {"name": "A", "delta": -1.0, "gamma": 0.2, "vega": 0.4},
            {"name": "B", "delta": -2.0, "gamma": 0.1, "vega": 0.7},
            {"name": "C", "delta": 0.5, "gamma": 0.3, "vega": 0.2},
        ],
        target_constraints={"delta": 0.0, "gamma": 0.0},
        factor_penalties={"vega": 5.0},
        risk_aversion=2.0,
        bounds=[(-100, 100), (-100, 100), (-100, 100)],
    )
    assert result["success"] is True
    assert abs(result["final_greeks"]["delta"]) < 1e-6
    assert abs(result["final_greeks"]["gamma"]) < 1e-6
