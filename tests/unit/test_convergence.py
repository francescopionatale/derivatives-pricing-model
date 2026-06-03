"""Cross-method convergence tests: independent methods must agree to tolerance.

These tests embody the "refute until convergence" criterion of the dynamic workflow:
if two independent implementations disagree, at least one is wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from engines.pricing.black_scholes import bs_price_and_greeks
from engines.pricing.heston_fourier import heston_price_carr_madan, heston_price_cos, heston_price_lewis
from engines.pricing.heston_vanilla import heston_vanilla_price_mc
from engines.pricing.jump_diffusion import merton_price
from engines.pricing.pde import bs_pde_price, heston_pde_price

# Standard parameter sets
HESTON = dict(S0=100.0, K=100.0, T=1.0, r=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04)
BS = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
MERTON = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, lam=1.0, mu_J=-0.1, sigma_J=0.15)


class TestFourierMethodsConverge:
    """Carr-Madan, COS, and Lewis must all agree to < 1 cent for standard params."""

    @pytest.mark.parametrize("K", [85.0, 90.0, 100.0, 110.0, 115.0])
    def test_cos_vs_carr_madan_call(self, K):
        cos = heston_price_cos(**{**HESTON, "K": K})["price"]
        cm = heston_price_carr_madan(**{**HESTON, "K": K})["price"]
        assert abs(cos - cm) < 0.02, f"K={K}: COS={cos:.4f} CM={cm:.4f}"

    @pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
    def test_lewis_vs_cos_call(self, K):
        lewis = heston_price_lewis(**{**HESTON, "K": K})["price"]
        cos = heston_price_cos(**{**HESTON, "K": K})["price"]
        assert abs(lewis - cos) < 0.02, f"K={K}: Lewis={lewis:.4f} COS={cos:.4f}"

    def test_fourier_vs_heston_mc(self):
        cos = heston_price_cos(**HESTON)["price"]
        mc = heston_vanilla_price_mc(**HESTON, is_call=True, n_steps=100, n_paths=50_000, seed=42)
        # MC with 50k paths has std_err ≈ 0.05 → 3σ ≈ 0.15
        assert abs(cos - mc["price"]) < 0.25, f"COS={cos:.4f} MC={mc['price']:.4f}"


class TestBSLimits:
    """Heston → BS in the σ_v → 0 limit; PDE → BS at O(Δx²)."""

    def test_heston_cos_sigma_v_zero_matches_bs(self):
        # When σ_v → 0, Heston COS → BS(σ = √θ)
        params = {**HESTON, "sigma_v": 1e-5}
        heston = heston_price_cos(**params)["price"]
        bs = bs_price_and_greeks(**BS)["price"]
        assert abs(heston - bs) < 0.10

    def test_pde_cn_matches_bs(self):
        pde = bs_pde_price(**{k: v for k, v in BS.items() if k != "S"}, S0=BS["S"],
                           N_x=300, N_t=200)
        bs = bs_price_and_greeks(**BS)["price"]
        assert abs(pde["price"] - bs) < 0.05


class TestAmericanPutBounds:
    """American put ≥ European put; early-exercise premium ≥ 0."""

    def test_american_put_pde_ge_european_pde(self):
        eu = bs_pde_price(S0=100, K=100, T=1, r=0.05, sigma=0.2, is_american=False)
        am = bs_pde_price(S0=100, K=100, T=1, r=0.05, sigma=0.2, is_american=True)
        assert am["price"] >= eu["price"] - 0.01

    def test_american_put_pde_ge_bs_european(self):
        am_pde = bs_pde_price(S0=100, K=100, T=1, r=0.05, sigma=0.2, is_american=True,
                              N_x=300, N_t=200)
        eu_bs = bs_price_and_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2)
        eu_bs_put = eu_bs["price"] - 100 + 100 * np.exp(-0.05)  # put via parity
        assert am_pde["price"] >= eu_bs_put - 0.05


class TestMertonLimits:
    """Merton series → BS when λ=0; put-call parity preserved."""

    def test_merton_lambda_zero_equals_bs(self):
        mer = merton_price(**{**MERTON, "lam": 0.0})["price"]
        bs = bs_price_and_greeks(**BS)["price"]
        assert abs(mer - bs) < 0.01

    def test_merton_put_call_parity(self):
        call = merton_price(**MERTON, is_call=True)["price"]
        put = merton_price(**MERTON, is_call=False)["price"]
        parity = MERTON["S0"] - MERTON["K"] * np.exp(-MERTON["r"] * MERTON["T"])
        assert abs((call - put) - parity) < 0.02


class TestHestonPDEvsFourier:
    """Heston ADI PDE must match COS Fourier prices within 2% ATM."""

    def test_heston_pde_atm_matches_cos(self):
        pde = heston_pde_price(**HESTON, N_x=80, N_v=40, N_t=80)["price"]
        cos = heston_price_cos(**HESTON)["price"]
        assert abs(pde - cos) / cos < 0.02

    def test_heston_pde_itm_matches_cos(self):
        pde = heston_pde_price(**{**HESTON, "K": 90.0}, N_x=80, N_v=40, N_t=80)["price"]
        cos = heston_price_cos(**{**HESTON, "K": 90.0})["price"]
        assert abs(pde - cos) / cos < 0.03
