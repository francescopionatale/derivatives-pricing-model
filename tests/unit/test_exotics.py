from quant_derivatives.engines.pricing.black_scholes import bs_price_and_greeks
from quant_derivatives.engines.pricing.exotics import (
    price_barrier_mc,
    price_barrier_heston_mc,
    price_lookback_mc,
    price_lookback_heston_mc,
)


def test_up_and_out_call_is_cheaper_than_vanilla_call_under_gbm():
    vanilla = bs_price_and_greeks(S=100, K=100, T=1.0, r=0.03, sigma=0.2, is_call=True)["price"]
    barrier = price_barrier_mc(
        S0=100,
        K=100,
        T=1.0,
        r=0.03,
        sigma=0.2,
        barrier=120,
        is_up=True,
        is_out=True,
        is_call=True,
        n_steps=126,
        n_paths=4000,
        seed=42,
        antithetic=True,
    )
    assert barrier["price"] > 0.0
    assert barrier["price"] < vanilla
    assert barrier["brownian_bridge_correction"] is True



def test_lookback_gbm_returns_positive_price_and_metadata():
    res = price_lookback_mc(
        S0=100,
        K=100,
        T=1.0,
        r=0.03,
        sigma=0.2,
        is_floating=False,
        is_call=True,
        n_steps=126,
        n_paths=3000,
        seed=7,
        antithetic=True,
    )
    assert res["price"] > 0.0
    assert len(res["ci_95"]) == 2
    assert res["lookback_style"] == "fixed"



def test_heston_barrier_and_lookback_run():
    barrier = price_barrier_heston_mc(
        S0=100,
        K=100,
        T=1.0,
        r=0.03,
        barrier=125,
        is_up=True,
        is_out=True,
        is_call=True,
        n_steps=100,
        n_paths=2000,
        kappa=2.0,
        theta=0.04,
        sigma_v=0.3,
        rho=-0.7,
        v0=0.04,
        seed=99,
    )
    lookback = price_lookback_heston_mc(
        S0=100,
        K=100,
        T=1.0,
        r=0.03,
        is_floating=True,
        is_call=True,
        n_steps=100,
        n_paths=2000,
        kappa=2.0,
        theta=0.04,
        sigma_v=0.3,
        rho=-0.7,
        v0=0.04,
        seed=99,
    )
    assert barrier["price"] >= 0.0
    assert lookback["price"] > 0.0
    assert barrier["model"] == "heston"
    assert lookback["model"] == "heston"
