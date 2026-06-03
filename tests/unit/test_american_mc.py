"""Tests for Longstaff-Schwartz American Monte Carlo pricing."""
from __future__ import annotations

import numpy as np
import pytest

from engines.pricing.american_mc import american_option_lsm
from engines.pricing.black_scholes import bs_price_and_greeks
from engines.pricing.pde import bs_pde_price

BASE = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


class TestLSMBasic:
    def test_returns_dict(self):
        res = american_option_lsm(**BASE, is_call=False, n_paths=200, n_steps=20, seed=42)
        for key in ("price", "std_err", "ci_low", "ci_hi", "method", "exercise_boundary"):
            assert key in res

    def test_method_tag(self):
        res = american_option_lsm(**BASE, is_call=False, n_paths=200, n_steps=20, seed=1)
        assert res["method"] == "lsm"

    def test_price_positive(self):
        res = american_option_lsm(**BASE, is_call=False, n_paths=500, n_steps=50, seed=7)
        assert res["price"] > 0.0

    def test_ci_contains_price(self):
        res = american_option_lsm(**BASE, is_call=False, n_paths=500, n_steps=50, seed=3)
        assert res["ci_low"] <= res["price"] <= res["ci_hi"]

    def test_exercise_boundary_shape(self):
        n_steps = 20
        res = american_option_lsm(**BASE, is_call=False, n_paths=300, n_steps=n_steps, seed=4)
        assert res["exercise_boundary"].shape == (n_steps,)

    def test_reproducible(self):
        kw = dict(**BASE, is_call=False, n_paths=300, n_steps=30, seed=99)
        p1 = american_option_lsm(**kw)["price"]
        p2 = american_option_lsm(**kw)["price"]
        assert p1 == p2


class TestLSMConvergence:
    def test_american_put_ge_european_put(self):
        # American put ≥ European put (early exercise premium ≥ 0)
        am = american_option_lsm(**BASE, is_call=False, n_paths=5000, n_steps=100, seed=42)
        eu = bs_price_and_greeks(S=BASE["S0"], K=BASE["K"], T=BASE["T"], r=BASE["r"], sigma=BASE["sigma"])
        # European put price
        eu_put = eu["price"] - BASE["S0"] + BASE["K"] * np.exp(-BASE["r"] * BASE["T"])
        assert am["price"] >= eu_put - 0.10  # allow 10c numerical tolerance

    def test_american_call_no_div_equals_european(self):
        # Without dividends, American call = European call (no early exercise)
        am = american_option_lsm(**BASE, is_call=True, n_paths=5000, n_steps=100, seed=42)
        eu = bs_price_and_greeks(S=BASE["S0"], K=BASE["K"], T=BASE["T"], r=BASE["r"], sigma=BASE["sigma"])
        # LSM call should be close to European call (early exercise is never optimal)
        assert abs(am["price"] - eu["price"]) < 0.30  # generous MC tolerance

    def test_lsm_matches_pde_american_put(self):
        # Cross-validate LSM (MC) vs PDE Crank-Nicolson American put
        pde = bs_pde_price(**BASE, is_call=False, is_american=True, N_x=300, N_t=200)
        lsm = american_option_lsm(**BASE, is_call=False, n_paths=20_000, n_steps=200, seed=42)
        # Tolerance: MC bias + PDE discretisation error ≈ 0.10
        assert abs(lsm["price"] - pde["price"]) < 0.20

    @pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
    def test_put_intrinsic_lower_bound(self, K):
        intrinsic = max(K - BASE["S0"], 0.0)
        res = american_option_lsm(**{**BASE, "K": K}, is_call=False, n_paths=2000, n_steps=50, seed=0)
        assert res["price"] >= intrinsic - 0.05
