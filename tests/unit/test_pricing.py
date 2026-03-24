import pytest
import numpy as np
from quant_derivatives.engines.pricing.black_scholes import bs_price_and_greeks
from quant_derivatives.engines.pricing.binomial import binomial_price

def test_bs_call_price():
    res = bs_price_and_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, is_call=True)
    assert np.isclose(res["price"], 10.45058, atol=1e-4)

def test_bs_put_price():
    res = bs_price_and_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, is_call=False)
    assert np.isclose(res["price"], 5.57352, atol=1e-4)

def test_binomial_converges_to_bs():
    bs_res = bs_price_and_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, is_call=True)
    bin_price = binomial_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2, n_steps=1000, is_call=True)
    assert np.isclose(bs_res["price"], bin_price, atol=1e-2)
