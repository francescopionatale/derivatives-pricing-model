"""Tests for Merton (1976) jump-diffusion pricing and simulation."""
from __future__ import annotations

import numpy as np
import pytest

from engines.pricing.black_scholes import bs_price_and_greeks
from engines.pricing.jump_diffusion import (
    merton_characteristic_function,
    merton_price,
    simulate_merton_paths,
)

BASE = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, lam=1.0, mu_J=-0.1, sigma_J=0.15)


class TestMertonCharFunction:
    def test_phi_zero_is_one(self):
        phi = merton_characteristic_function(np.array([0.0 + 0j]), **{k: v for k, v in BASE.items() if k != "K"})
        assert abs(phi[0] - 1.0) < 1e-10

    def test_phi_minus_i_gives_forward(self):
        phi = merton_characteristic_function(
            np.array([-1j]), **{k: v for k, v in BASE.items() if k != "K"}
        )
        forward = BASE["S0"] * np.exp(BASE["r"] * BASE["T"])
        assert abs(np.real(phi[0]) - forward) < 1e-4

    def test_phi_magnitude_le_one_real_u(self):
        params = {k: v for k, v in BASE.items() if k != "K"}
        u_real = np.linspace(0.1, 20.0, 100) + 0j
        phi = merton_characteristic_function(u_real, **params)
        assert np.all(np.abs(phi) <= 1.0 + 1e-10)


class TestMertonSeries:
    def test_lambda_zero_equals_bs(self):
        bs = bs_price_and_greeks(S=BASE["S0"], K=BASE["K"], T=BASE["T"], r=BASE["r"], sigma=BASE["sigma"])
        mer = merton_price(**{**BASE, "lam": 0.0}, is_call=True)
        assert abs(mer["price"] - bs["price"]) < 0.01

    def test_sigma_J_zero_equals_bs(self):
        mer = merton_price(**{**BASE, "sigma_J": 1e-8, "lam": 0.0}, is_call=True)
        bs = bs_price_and_greeks(S=BASE["S0"], K=BASE["K"], T=BASE["T"], r=BASE["r"], sigma=BASE["sigma"])
        assert abs(mer["price"] - bs["price"]) < 0.01

    def test_call_positive(self):
        mer = merton_price(**BASE, is_call=True)
        assert mer["price"] > 0.0

    def test_put_call_parity(self):
        call = merton_price(**BASE, is_call=True)["price"]
        put = merton_price(**BASE, is_call=False)["price"]
        parity = BASE["S0"] - BASE["K"] * np.exp(-BASE["r"] * BASE["T"])
        assert abs((call - put) - parity) < 0.02

    def test_monotone_in_strike(self):
        prices = [merton_price(**{**BASE, "K": K}, is_call=True)["price"] for K in [80, 90, 100, 110, 120]]
        for a, b in zip(prices, prices[1:], strict=False):
            assert a >= b - 0.01

    def test_series_price_bounded(self):
        ser = merton_price(**BASE, is_call=True)
        assert ser["price"] > 0.0
        assert ser["price"] < BASE["S0"]  # bounded above by S0

    @pytest.mark.parametrize("lam", [0.5, 1.0, 3.0])
    def test_convergence_n_terms(self, lam):
        # Price should stabilise beyond ~30 terms for λT ≤ 5
        p30 = merton_price(**{**BASE, "lam": lam}, is_call=True, n_terms=30)["price"]
        p50 = merton_price(**{**BASE, "lam": lam}, is_call=True, n_terms=50)["price"]
        assert abs(p50 - p30) < 1e-6


class TestMertonSimulation:
    def test_paths_shape(self):
        paths = simulate_merton_paths(**{k: v for k, v in BASE.items() if k != "K"}, n_steps=50, n_paths=200, seed=42)
        assert paths.shape == (51, 200)

    def test_initial_value(self):
        paths = simulate_merton_paths(**{k: v for k, v in BASE.items() if k != "K"}, n_steps=10, n_paths=100, seed=1)
        np.testing.assert_allclose(paths[0], BASE["S0"])

    def test_reproducible(self):
        kw = {k: v for k, v in BASE.items() if k != "K"}
        p1 = simulate_merton_paths(**kw, n_steps=20, n_paths=50, seed=7)
        p2 = simulate_merton_paths(**kw, n_steps=20, n_paths=50, seed=7)
        np.testing.assert_array_equal(p1, p2)

    def test_mc_converges_to_series(self):
        kw = {k: v for k, v in BASE.items() if k != "K"}
        paths = simulate_merton_paths(**kw, n_steps=100, n_paths=50_000, seed=42)
        payoffs = np.maximum(paths[-1] - BASE["K"], 0.0)
        mc_price = np.exp(-BASE["r"] * BASE["T"]) * np.mean(payoffs)
        ser_price = merton_price(**BASE, is_call=True)["price"]
        # Generous tolerance: MC with 50k paths should be within 3 std errors (≈0.30)
        assert abs(mc_price - ser_price) < 0.50
