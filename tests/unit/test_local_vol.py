"""Tests for local volatility (Dupire) and SVI calibration."""
from __future__ import annotations

import numpy as np
import pytest

from engines.pricing.local_vol import calibrate_svi, check_svi_arbitrage, dupire_local_vol


class TestDupireLocalVol:
    def _flat_surface(self, sigma: float = 0.2) -> object:
        def iv(K: float, T: float) -> float:
            return sigma
        return iv

    def test_flat_surface_returns_sigma(self):
        sigma = 0.2
        res = dupire_local_vol(K=100.0, T=1.0, S0=100.0, r=0.05, iv_surface=self._flat_surface(sigma))
        # For a flat surface, local vol ≈ implied vol
        assert abs(res["local_vol"] - sigma) < 0.02

    def test_local_vol_nonnegative(self):
        # Even for non-monotone inputs, local vol must be clamped ≥ 0
        def noisy(K: float, T: float) -> float:
            return 0.2 + 0.01 * np.sin(K)
        for K in [80.0, 100.0, 120.0]:
            res = dupire_local_vol(K=K, T=1.0, S0=100.0, r=0.05, iv_surface=noisy)
            assert res["local_vol"] >= 0.0

    def test_returns_dict_keys(self):
        res = dupire_local_vol(K=100.0, T=1.0, S0=100.0, r=0.05, iv_surface=self._flat_surface())
        assert "local_vol" in res and "method" in res


class TestSVICalibration:
    def _make_svi_data(self, a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.3):
        from engines.pricing.local_vol import _svi_total_var
        k = np.linspace(-1.0, 1.0, 20)
        w = _svi_total_var(k, a, b, rho, m, sigma)
        return k, w

    def test_recovers_svi_params(self):
        k, w = self._make_svi_data(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.3)
        result = calibrate_svi(k, w)
        assert result["rmse"] < 1e-3
        p = result["params"]
        assert abs(p["a"] - 0.04) < 0.01
        assert abs(p["b"] - 0.10) < 1e-3
        assert abs(p["rho"] - (-0.3)) < 1e-3

    def test_fitted_total_var_nonnegative(self):
        k, w = self._make_svi_data()
        result = calibrate_svi(k, w)
        assert np.all(result["fitted_total_var"] >= 0.0)

    def test_returns_dict_keys(self):
        k, w = self._make_svi_data()
        result = calibrate_svi(k, w)
        for key in ("params", "rmse", "method", "fitted_total_var"):
            assert key in result

    @pytest.mark.parametrize("rho", [-0.5, 0.0, 0.5])
    def test_fit_various_rho(self, rho):
        k, w = self._make_svi_data(rho=rho)
        result = calibrate_svi(k, w)
        assert result["rmse"] < 0.01


class TestSVIArbitrage:
    def _params_no_arb(self) -> dict:
        return {"a": 0.04, "b": 0.04, "rho": 0.0, "m": 0.0, "sigma": 0.5}

    def test_no_arbitrage_for_mild_params(self):
        result = check_svi_arbitrage(self._params_no_arb())
        assert result["butterfly_ok"]
        assert result["min_g"] >= -1e-8

    def test_returns_g_array(self):
        result = check_svi_arbitrage(self._params_no_arb())
        assert "g" in result
        assert len(result["g"]) == len(result["k_grid"])

    def test_custom_k_grid(self):
        k_grid = np.linspace(-2.0, 2.0, 100)
        result = check_svi_arbitrage(self._params_no_arb(), k_grid=k_grid)
        assert len(result["g"]) == 100
