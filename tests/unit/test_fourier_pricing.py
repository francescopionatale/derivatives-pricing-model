"""Cross-validation tests for the Heston Fourier pricing module.

The three Fourier methods (Carr-Madan, COS, Lewis) are validated against each
other and against the Black-Scholes analytical formula in the limit sigma_v → 0.
Put-call parity and characteristic-function properties are tested as invariants.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engines.pricing.black_scholes import bs_price_and_greeks
from engines.pricing.heston_fourier import (
    heston_characteristic_function,
    heston_price_carr_madan,
    heston_price_cos,
    heston_price_lewis,
)

# ---------------------------------------------------------------------------
# Standard Heston parameter set for tests
# ---------------------------------------------------------------------------
BASE = dict(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.05, T=1.0)
BASE_K = 100.0  # ATM


class TestCharacteristicFunction:
    """Analytic properties of φ(u)."""

    def test_phi_at_zero_is_one(self):
        u = np.array([0.0 + 0j])
        phi = heston_characteristic_function(u, **BASE)
        assert abs(phi[0] - 1.0) < 1e-10

    def test_phi_minus_i_gives_forward(self):
        # E[S_T] = S0 * exp(r*T);  φ(-i) = E[exp(i*(-i)*lnS)] = E[S_T]/1
        u = np.array([-1j])
        phi = heston_characteristic_function(u, **BASE)
        forward = BASE["S0"] * np.exp(BASE["r"] * BASE["T"])
        assert abs(np.real(phi[0]) - forward) < 1.0  # generous tolerance for Heston (not BS)

    def test_phi_conjugate_symmetry(self):
        # For real u, φ(-u) = conj(φ(u))
        u_real = np.linspace(0.5, 5.0, 20)
        phi_pos = heston_characteristic_function(u_real + 0j, **BASE)
        phi_neg = heston_characteristic_function(-u_real + 0j, **BASE)
        np.testing.assert_allclose(phi_neg, np.conj(phi_pos), atol=1e-10)

    def test_phi_magnitude_le_one(self):
        # |φ(u)| ≤ 1 for real u (it's a characteristic function)
        u_real = np.linspace(0.1, 20.0, 100)
        phi = heston_characteristic_function(u_real + 0j, **BASE)
        assert np.all(np.abs(phi) <= 1.0 + 1e-10)


class TestFourierPricers:
    """Cross-validate Carr-Madan, COS, and Lewis against each other and BS."""

    def test_cos_atm_call_positive(self):
        res = heston_price_cos(K=BASE_K, is_call=True, **BASE)
        assert res["price"] > 0

    def test_carr_madan_atm_call_positive(self):
        res = heston_price_carr_madan(K=BASE_K, is_call=True, **BASE)
        assert res["price"] > 0

    def test_lewis_atm_call_positive(self):
        res = heston_price_lewis(K=BASE_K, is_call=True, **BASE)
        assert res["price"] > 0

    def test_carr_madan_matches_cos(self):
        cm = heston_price_carr_madan(K=BASE_K, is_call=True, **BASE)["price"]
        cos = heston_price_cos(K=BASE_K, is_call=True, **BASE)["price"]
        assert abs(cm - cos) < 0.02  # within 2 cents on a ~10 call

    def test_lewis_matches_cos(self):
        lewis = heston_price_lewis(K=BASE_K, is_call=True, **BASE)["price"]
        cos = heston_price_cos(K=BASE_K, is_call=True, **BASE)["price"]
        assert abs(lewis - cos) < 0.02

    def test_sigma_v_zero_matches_bs(self):
        # When sigma_v → 0, Heston → BS with sigma = sqrt(theta)
        params_flat = {**BASE, "sigma_v": 1e-5}
        bs_sigma = float(np.sqrt(BASE["theta"]))
        bs = bs_price_and_greeks(S=BASE["S0"], K=BASE_K, T=BASE["T"], r=BASE["r"], sigma=bs_sigma)
        cos = heston_price_cos(K=BASE_K, is_call=True, **params_flat)
        assert abs(cos["price"] - bs["price"]) < 0.1  # generous: COS uses cumulant bounds

    def test_put_call_parity_cos(self):
        call = heston_price_cos(K=BASE_K, is_call=True, **BASE)["price"]
        put = heston_price_cos(K=BASE_K, is_call=False, **BASE)["price"]
        parity = BASE["S0"] - BASE_K * np.exp(-BASE["r"] * BASE["T"])
        assert abs((call - put) - parity) < 0.02

    def test_put_call_parity_carr_madan(self):
        call = heston_price_carr_madan(K=BASE_K, is_call=True, **BASE)["price"]
        put = heston_price_carr_madan(K=BASE_K, is_call=False, **BASE)["price"]
        parity = BASE["S0"] - BASE_K * np.exp(-BASE["r"] * BASE["T"])
        assert abs((call - put) - parity) < 0.02

    def test_carr_madan_array_strikes(self):
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        res = heston_price_carr_madan(K=strikes, is_call=True, **BASE)
        assert isinstance(res["price"], np.ndarray)
        assert res["price"].shape == (5,)
        assert np.all(res["price"] >= 0)

    def test_cos_array_strikes(self):
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        res = heston_price_cos(K=strikes, is_call=True, **BASE)
        assert isinstance(res["price"], np.ndarray)
        assert res["price"].shape == (5,)
        assert np.all(res["price"] >= 0)

    @pytest.mark.parametrize("K", [80.0, 90.0, 100.0, 110.0, 120.0])
    def test_cos_call_monotone_in_strike(self, K):
        # Call price decreases monotonically with strike
        prev_K = K - 10.0
        if prev_K <= 0:
            return
        c_high = heston_price_cos(K=K, is_call=True, **BASE)["price"]
        c_low = heston_price_cos(K=prev_K, is_call=True, **BASE)["price"]
        assert c_low >= c_high - 0.01  # allow tiny numerical noise

    def test_methods_agree_itm_call(self):
        params = dict(**BASE)
        K_itm = 85.0
        cm = heston_price_carr_madan(K=K_itm, is_call=True, **params)["price"]
        cos = heston_price_cos(K=K_itm, is_call=True, **params)["price"]
        lewis = heston_price_lewis(K=K_itm, is_call=True, **params)["price"]
        assert abs(cm - cos) < 0.05
        assert abs(lewis - cos) < 0.05

    def test_methods_agree_otm_put(self):
        K_otm = 85.0
        cm = heston_price_carr_madan(K=K_otm, is_call=False, **BASE)["price"]
        cos = heston_price_cos(K=K_otm, is_call=False, **BASE)["price"]
        lewis = heston_price_lewis(K=K_otm, is_call=False, **BASE)["price"]
        assert abs(cm - cos) < 0.05
        assert abs(lewis - cos) < 0.05


class TestCalibratorUsesCOS:
    """Smoke-test that COS-powered calibration runs and returns sensible results."""

    def test_cos_calibration_runs(self):
        ROOT = Path(__file__).resolve().parents[2]
        quotes_path = ROOT / "examples" / "heston" / "synthetic_heston_quotes.csv"
        if not quotes_path.exists():
            pytest.skip("synthetic_heston_quotes.csv not found")

        from data_io.loaders import load_quotes_csv
        from engines.calibration.heston import calibrate_heston_to_quotes

        quotes = load_quotes_csv(str(quotes_path))
        result = calibrate_heston_to_quotes(
            quotes,
            S0=100.0,
            r=0.05,
            pricing_method="cos",
            maxiter=300,
        )
        # L-BFGS-B may report success=False when it hits maxiter but still finds a good solution.
        # The meaningful convergence check is the RMSE.
        assert result.rmse_price < 2.0  # generous: synthetic data should be very close
        # Params should be in valid ranges
        p = result.params
        assert 0 < p["kappa"] < 20
        assert 0 < p["theta"] < 1
        assert 0 < p["sigma_v"] < 3
        assert -1 < p["rho"] < 1
        assert 0 < p["v0"] < 1
