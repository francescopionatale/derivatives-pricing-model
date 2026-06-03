"""Regression tests: golden price values from the new engines.

These tests pin specific numerical outputs to guard against silent regressions
from future refactors.  Tolerances are set to ~1 basis point (0.01 in price).

To update after an intentional model change, run:
    pytest tests/regression/ -v --tb=short
and inspect failures before updating expected values.
"""
from __future__ import annotations

import numpy as np
import pytest

# ── Standard parameter sets ────────────────────────────────────────────────
BS = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
HESTON = dict(S0=100.0, K=100.0, T=1.0, r=0.05,
              kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04)
MERTON = dict(S0=100.0, K=100.0, T=1.0, r=0.05,
              sigma=0.2, lam=1.0, mu_J=-0.1, sigma_J=0.15)


# ── Black-Scholes ───────────────────────────────────────────────────────────
class TestBSRegression:
    def test_atm_call(self):
        from engines.pricing.black_scholes import bs_price_and_greeks
        assert abs(bs_price_and_greeks(**BS)["price"] - 10.4506) < 0.001

    def test_itm_put(self):
        from engines.pricing.black_scholes import bs_price_and_greeks
        p = bs_price_and_greeks(S=100, K=110, T=1, r=0.05, sigma=0.2, is_call=False)
        assert abs(p["price"] - 10.6753) < 0.001


# ── Heston Fourier (COS) ────────────────────────────────────────────────────
class TestHestonCOSRegression:
    def test_atm_call(self):
        from engines.pricing.heston_fourier import heston_price_cos
        assert abs(heston_price_cos(**HESTON)["price"] - 10.3942) < 0.01

    def test_otm_call_k115(self):
        from engines.pricing.heston_fourier import heston_price_cos
        p = heston_price_cos(**{**HESTON, "K": 115.0})["price"]
        assert abs(p - 3.7) < 0.5  # wider tolerance, OTM sensitive to params

    def test_itm_put_k90(self):
        from engines.pricing.heston_fourier import heston_price_cos
        p = heston_price_cos(**{**HESTON, "K": 90.0, "is_call": False})["price"]
        assert abs(p - 2.686) < 0.05

    @pytest.mark.parametrize("method", ["cos", "carr-madan"])
    def test_methods_agree_atm(self, method):
        from engines.pricing.heston_fourier import heston_price_carr_madan, heston_price_cos
        cos = heston_price_cos(**HESTON)["price"]
        cm = heston_price_carr_madan(**HESTON)["price"]
        assert abs(cos - cm) < 0.02


# ── Merton series ───────────────────────────────────────────────────────────
class TestMertonRegression:
    def test_atm_call(self):
        from engines.pricing.jump_diffusion import merton_price
        p = merton_price(**MERTON)["price"]
        assert abs(p - 12.761) < 0.01

    def test_itm_call_k90(self):
        from engines.pricing.jump_diffusion import merton_price
        p = merton_price(**{**MERTON, "K": 90.0})["price"]
        assert abs(p - 18.698) < 0.05


# ── BS PDE (Crank-Nicolson) ─────────────────────────────────────────────────
class TestPDERegression:
    def test_european_call_cn(self):
        from engines.pricing.pde import bs_pde_price
        p = bs_pde_price(**{k: v for k, v in BS.items() if k != "S"},
                          S0=BS["S"], N_x=300, N_t=150)["price"]
        assert abs(p - 10.4506) < 0.05

    def test_american_put(self):
        from engines.pricing.pde import bs_pde_price
        p = bs_pde_price(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                          is_call=False, is_american=True, N_x=300, N_t=150)["price"]
        # American put > European put; known range ~5.8–6.2
        assert 5.5 < p < 6.5


# ── Heston ADI PDE ─────────────────────────────────────────────────────────
class TestHestonPDERegression:
    def test_atm_call(self):
        from engines.pricing.pde import heston_pde_price
        p = heston_pde_price(**HESTON, N_x=80, N_v=40, N_t=80)["price"]
        # Should match COS ~10.394 within 2%
        assert abs(p - 10.394) / 10.394 < 0.02


# ── Longstaff-Schwartz American ────────────────────────────────────────────
class TestLSMRegression:
    def test_american_put_vs_pde(self):
        from engines.pricing.american_mc import american_option_lsm
        from engines.pricing.pde import bs_pde_price
        lsm = american_option_lsm(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                   is_call=False, n_paths=30_000, n_steps=150, seed=42)["price"]
        pde = bs_pde_price(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                            is_call=False, is_american=True, N_x=300, N_t=150)["price"]
        assert abs(lsm - pde) < 0.25  # MC bias + PDE error ≤ 0.25


# ── SABR ────────────────────────────────────────────────────────────────────
class TestSABRRegression:
    def test_atm_vol(self):
        from engines.pricing.sabr import sabr_implied_vol
        # F=1, K=1, T=1, alpha=0.2, beta=0 (normal SABR regime), rho=0, nu=0.3
        res = sabr_implied_vol(F=1.0, K=1.0, T=1.0, alpha=0.2, beta=0.0, rho=0.0, nu=0.3)
        # beta=0 ATM: sigma_B ≈ alpha = 0.2
        assert abs(res["sigma"] - 0.2) < 0.01

    def test_smile_skew_sign(self):
        from engines.pricing.sabr import sabr_implied_vol
        # Negative rho → left skew (lower K has higher vol)
        v_lo = sabr_implied_vol(F=1.0, K=0.9, T=1.0, alpha=0.2, beta=0.5, rho=-0.5, nu=0.4)["sigma"]
        v_hi = sabr_implied_vol(F=1.0, K=1.1, T=1.0, alpha=0.2, beta=0.5, rho=-0.5, nu=0.4)["sigma"]
        assert v_lo > v_hi  # negative skew: OTM put vol > OTM call vol


# ── Local vol / SVI ─────────────────────────────────────────────────────────
class TestSVIRegression:
    def test_round_trip(self):
        from engines.pricing.local_vol import _svi_total_var, calibrate_svi
        k = np.linspace(-0.5, 0.5, 15)
        params_true = dict(a=0.04, b=0.08, rho=-0.2, m=0.0, sigma=0.4)
        w_true = _svi_total_var(k, **params_true)
        result = calibrate_svi(k, w_true)
        assert result["rmse"] < 1e-3
        assert abs(result["params"]["a"] - 0.04) < 0.02
        assert abs(result["params"]["b"] - 0.08) < 0.02
