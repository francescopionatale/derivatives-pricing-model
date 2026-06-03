"""Pricing invariants: put-call parity, monotonicity, boundary conditions.

These tests verify fundamental no-arbitrage properties across ALL implemented engines.
Any failure represents a genuine numerical or logic bug.
"""
from __future__ import annotations

import numpy as np
import pytest

from engines.pricing.black_scholes import bs_price_and_greeks
from engines.pricing.heston_fourier import heston_price_carr_madan, heston_price_cos
from engines.pricing.jump_diffusion import merton_price
from engines.pricing.pde import bs_pde_price
from engines.pricing.sabr import sabr_implied_vol

HESTON = dict(S0=100.0, K=100.0, T=1.0, r=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04)


class TestPutCallParity:
    """C − P = S₀ − K·e^{−rT} for all European pricers."""

    def _parity(self, S0, K, r, T):
        return S0 - K * np.exp(-r * T)

    def test_parity_bs(self):
        call = bs_price_and_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2)["price"]
        put = bs_price_and_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2, is_call=False)["price"]
        assert abs((call - put) - self._parity(100, 100, 0.05, 1)) < 1e-6

    def test_parity_heston_cos(self):
        call = heston_price_cos(**HESTON)["price"]
        put = heston_price_cos(**{**HESTON, "is_call": False})["price"]
        assert abs((call - put) - self._parity(100, 100, 0.05, 1)) < 0.02

    def test_parity_heston_cm(self):
        call = heston_price_carr_madan(**HESTON)["price"]
        put = heston_price_carr_madan(**{**HESTON, "is_call": False})["price"]
        assert abs((call - put) - self._parity(100, 100, 0.05, 1)) < 0.02

    def test_parity_merton(self):
        MERTON = dict(S0=100, K=100, T=1, r=0.05, sigma=0.2, lam=1.0, mu_J=-0.1, sigma_J=0.15)
        call = merton_price(**MERTON, is_call=True)["price"]
        put = merton_price(**MERTON, is_call=False)["price"]
        assert abs((call - put) - self._parity(100, 100, 0.05, 1)) < 0.02

    def test_parity_pde(self):
        kw = dict(S0=100, K=100, T=1, r=0.05, sigma=0.2, N_x=200, N_t=100)
        call = bs_pde_price(**kw, is_call=True)["price"]
        put = bs_pde_price(**kw, is_call=False)["price"]
        assert abs((call - put) - self._parity(100, 100, 0.05, 1)) < 0.05


class TestCallMonotonicity:
    """Call prices decrease monotonically with strike."""

    @pytest.mark.parametrize("engine", ["cos", "cm"])
    def test_heston_call_decreasing(self, engine):
        strikes = [80, 90, 100, 110, 120]
        if engine == "cos":
            prices = [heston_price_cos(**{**HESTON, "K": float(K)})["price"] for K in strikes]
        else:
            prices = [heston_price_carr_madan(**{**HESTON, "K": float(K)})["price"] for K in strikes]
        for a, b in zip(prices, prices[1:], strict=False):
            assert a >= b - 0.01

    def test_bs_call_decreasing(self):
        prices = [bs_price_and_greeks(S=100, K=float(K), T=1, r=0.05, sigma=0.2)["price"]
                  for K in [80, 90, 100, 110, 120]]
        for a, b in zip(prices, prices[1:], strict=False):
            assert a >= b - 1e-6


class TestPricePositivity:
    """All option prices must be ≥ 0."""

    @pytest.mark.parametrize("K", [70, 80, 90, 100, 110, 120, 130])
    def test_heston_cos_nonnegative(self, K):
        assert heston_price_cos(**{**HESTON, "K": float(K)})["price"] >= 0.0
        assert heston_price_cos(**{**HESTON, "K": float(K), "is_call": False})["price"] >= 0.0

    @pytest.mark.parametrize("K", [80, 100, 120])
    def test_merton_nonnegative(self, K):
        MERTON = dict(S0=100, K=float(K), T=1, r=0.05, sigma=0.2, lam=1.0, mu_J=-0.1, sigma_J=0.15)
        assert merton_price(**MERTON, is_call=True)["price"] >= 0.0
        assert merton_price(**MERTON, is_call=False)["price"] >= 0.0


class TestGreeks:
    """BS delta must be in [0, 1] for calls and [-1, 0] for puts."""

    def test_bs_call_delta_in_range(self):
        for K in [80, 90, 100, 110, 120]:
            res = bs_price_and_greeks(S=100, K=float(K), T=1, r=0.05, sigma=0.2)
            assert 0.0 <= res["delta"] <= 1.0

    def test_bs_put_delta_in_range(self):
        for K in [80, 90, 100, 110, 120]:
            res = bs_price_and_greeks(S=100, K=float(K), T=1, r=0.05, sigma=0.2, is_call=False)
            assert -1.0 <= res["delta"] <= 0.0

    def test_bs_gamma_positive(self):
        res = bs_price_and_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2)
        assert res["gamma"] > 0.0


class TestSABRInvariants:
    """SABR vol must be positive; Hagan formula is C¹ in strike."""

    def test_sabr_vol_positive(self):
        for K in [0.8, 0.9, 1.0, 1.1, 1.2]:
            res = sabr_implied_vol(F=1.0, K=K, T=1.0, alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
            assert res["sigma"] > 0.0

    def test_sabr_vol_smooth_in_strike(self):
        strikes = np.linspace(0.7, 1.3, 20)
        vols = [sabr_implied_vol(F=1.0, K=K, T=1.0, alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)["sigma"]
                for K in strikes]
        # Smoothness: no jump > 0.05 between adjacent strikes
        for a, b in zip(vols, vols[1:], strict=False):
            assert abs(b - a) < 0.05
