import numpy as np
from quant_derivatives.engines.simulation.gbm import simulate_gbm_paths
from quant_derivatives.engines.hedging.discrete_hedging import simulate_discrete_hedging


def test_hedging_pnl_mean_near_zero():
    # In a Black-Scholes world with no transaction costs, the mean P&L should be near zero
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    n_steps, n_paths = 100, 500

    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=42)
    result = simulate_discrete_hedging(paths, K, T, r, sigma, is_call=True, transaction_cost=0.0)
    pnls = result["total_pnl"]

    assert abs(np.mean(pnls)) < 0.5
