"""Tests for Heston 2-D ADI PDE pricing — cross-validates against Fourier (COS)."""
from __future__ import annotations

import numpy as np
import pytest

from engines.pricing.heston_fourier import heston_price_cos
from engines.pricing.pde import heston_pde_price

BASE = dict(S0=100.0, K=100.0, T=1.0, r=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04)
PDE_KW = dict(N_x=80, N_v=40, N_t=80)


class TestHestonPDEBasic:
    def test_returns_dict_keys(self):
        res = heston_pde_price(**BASE, **PDE_KW)
        for key in ("price", "method", "grid_S", "grid_v", "grid_V"):
            assert key in res

    def test_method_tag(self):
        assert heston_pde_price(**BASE, **PDE_KW)["method"] == "heston-pde-adi"

    def test_price_positive(self):
        assert heston_pde_price(**BASE, **PDE_KW)["price"] > 0.0

    def test_put_call_parity(self):
        call = heston_pde_price(**BASE, **PDE_KW, is_call=True)["price"]
        put = heston_pde_price(**BASE, **PDE_KW, is_call=False)["price"]
        parity = BASE["S0"] - BASE["K"] * np.exp(-BASE["r"] * BASE["T"])
        assert abs((call - put) - parity) < 0.30  # PDE has discretisation error

    def test_grid_shapes(self):
        res = heston_pde_price(**BASE, N_x=40, N_v=20, N_t=40)
        assert res["grid_S"].shape == (41,)
        assert res["grid_v"].shape == (21,)
        assert res["grid_V"].shape == (41, 21)


class TestHestonPDEvsFourier:
    def test_atm_call_matches_cos(self):
        pde = heston_pde_price(**BASE, **PDE_KW)["price"]
        cos = heston_price_cos(**BASE)["price"]
        assert abs(pde - cos) / cos < 0.015  # within 1.5%

    def test_itm_call_matches_cos(self):
        params = {**BASE, "K": 90.0}
        pde = heston_pde_price(**params, **PDE_KW)["price"]
        cos = heston_price_cos(**params)["price"]
        assert abs(pde - cos) / cos < 0.02

    def test_otm_call_matches_cos(self):
        params = {**BASE, "K": 110.0}
        pde = heston_pde_price(**params, **PDE_KW)["price"]
        cos = heston_price_cos(**params)["price"]
        assert abs(pde - cos) < 0.30  # OTM harder due to boundary effects

    @pytest.mark.parametrize("K", [85.0, 95.0, 100.0, 105.0, 115.0])
    def test_monotone_in_strike(self, K):
        K_lo = K - 5.0
        p_lo = heston_pde_price(**{**BASE, "K": K_lo}, **PDE_KW)["price"]
        p_hi = heston_pde_price(**{**BASE, "K": K}, **PDE_KW)["price"]
        assert p_lo >= p_hi - 0.10  # call decreases with strike

    def test_price_nonnegative_all_strikes(self):
        for K in [70, 80, 90, 100, 110, 120, 130]:
            res = heston_pde_price(**{**BASE, "K": float(K)}, N_x=60, N_v=30, N_t=60)
            assert res["price"] >= 0.0
