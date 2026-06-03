"""Unit tests for the SABR model — Hagan (2002) and calibration."""

import numpy as np
import pytest

from engines.pricing.sabr import (
    calibrate_sabr,
    sabr_implied_vol,
    sabr_normal_vol,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_smile(
    F: float,
    T: float,
    strikes: np.ndarray,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    return np.array(
        [sabr_implied_vol(F, K, T, alpha, beta, rho, nu)["sigma"] for K in strikes],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# TestSABRImpliedVol
# ---------------------------------------------------------------------------

class TestSABRImpliedVol:
    def test_atm_vol_nonzero(self):
        """ATM case returns a positive implied vol."""
        res = sabr_implied_vol(F=100, K=100, T=1, alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)
        assert res["sigma"] > 0.0

    def test_smile_skew(self):
        """Negative rho produces negative skew: OTM puts (K < F) > OTM calls (K > F)."""
        F, T = 100.0, 1.0
        alpha, beta, rho, nu = 0.3, 0.5, -0.3, 0.4
        iv_put = sabr_implied_vol(F, K=90.0, T=T, alpha=alpha, beta=beta, rho=rho, nu=nu)["sigma"]
        iv_call = sabr_implied_vol(F, K=110.0, T=T, alpha=alpha, beta=beta, rho=rho, nu=nu)["sigma"]
        assert iv_put > iv_call

    def test_beta_one_limit(self):
        """beta=1 (lognormal SABR) returns a positive vol."""
        res = sabr_implied_vol(F=100, K=100, T=1, alpha=0.3, beta=1.0, rho=-0.3, nu=0.4)
        assert res["sigma"] > 0.0

    def test_beta_zero(self):
        """beta=0 (normal SABR backbone) returns a positive lognormal vol."""
        res = sabr_implied_vol(F=100, K=100, T=1, alpha=0.3, beta=0.0, rho=-0.3, nu=0.4)
        assert res["sigma"] > 0.0

    def test_atm_case_matches_approx(self):
        """ATM branch (K=F) vs near-ATM (K=F+1e-8) agree within 1e-6."""
        F, T = 100.0, 1.0
        alpha, beta, rho, nu = 0.3, 0.5, -0.3, 0.4
        sigma_atm = sabr_implied_vol(F, K=F, T=T, alpha=alpha, beta=beta, rho=rho, nu=nu)["sigma"]
        sigma_near = sabr_implied_vol(F, K=F + 1e-8, T=T, alpha=alpha, beta=beta, rho=rho, nu=nu)["sigma"]
        assert abs(sigma_atm - sigma_near) < 1e-6

    def test_nu_zero_gives_black_vol(self):
        """nu=0 (flat vol): implied vol is roughly constant across strikes and equals
        alpha/F^(1-beta) at the money."""
        F, T, alpha, beta = 100.0, 1.0, 0.3, 0.5
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vols = _make_smile(F, T, strikes, alpha=alpha, beta=beta, rho=0.0, nu=0.0)
        # Spread of smile should be very small when nu=0
        assert (vols.max() - vols.min()) < 0.01
        # ATM vol should be close to alpha / F^(1-beta)
        expected_atm = alpha / F ** (1.0 - beta)
        assert abs(vols[2] - expected_atm) < 0.01

    def test_model_key_in_result(self):
        """Result dict contains the expected keys."""
        res = sabr_implied_vol(F=100, K=100, T=1, alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)
        assert "sigma" in res
        assert res["model"] == "sabr"
        assert "correction" in res

    def test_obloj_correction_alias(self):
        """obloj correction is accepted and returns a positive vol."""
        res = sabr_implied_vol(
            F=100, K=105, T=1, alpha=0.3, beta=0.5, rho=-0.3, nu=0.4,
            correction="obloj",
        )
        assert res["sigma"] > 0.0
        assert res["correction"] == "obloj"

    def test_alpha_zero_returns_zero(self):
        """alpha=0 should return sigma=0 without raising."""
        res = sabr_implied_vol(F=100, K=100, T=1, alpha=0.0, beta=0.5, rho=0.0, nu=0.4)
        assert res["sigma"] == 0.0

    def test_extreme_rho_guard(self):
        """rho clipped to ±0.9999 — should not raise even with rho=1."""
        res = sabr_implied_vol(F=100, K=100, T=1, alpha=0.3, beta=0.5, rho=0.9999, nu=0.4)
        assert np.isfinite(res["sigma"])

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            sabr_implied_vol(F=100, K=100, T=1, alpha=0.3, beta=1.5, rho=0.0, nu=0.4)

    def test_invalid_F_raises(self):
        with pytest.raises(ValueError):
            sabr_implied_vol(F=-100, K=100, T=1, alpha=0.3, beta=0.5, rho=0.0, nu=0.4)


# ---------------------------------------------------------------------------
# TestSABRNormalVol
# ---------------------------------------------------------------------------

class TestSABRNormalVol:
    def test_normal_vol_positive(self):
        """Returns a positive sigma_normal for a standard OTM case."""
        res = sabr_normal_vol(F=100.0, K=95.0, T=1.0, alpha=0.3, rho=-0.3, nu=0.4)
        assert res["sigma_normal"] > 0.0

    def test_atm_normal_vol(self):
        """F=K (ATM) should not raise and should return a positive vol."""
        res = sabr_normal_vol(F=100.0, K=100.0, T=1.0, alpha=0.3, rho=-0.3, nu=0.4)
        assert res["sigma_normal"] > 0.0

    def test_model_key(self):
        res = sabr_normal_vol(F=100.0, K=100.0, T=1.0, alpha=0.3, rho=0.0, nu=0.4)
        assert res["model"] == "sabr-normal"
        assert "sigma_normal" in res

    def test_atm_limit_continuity(self):
        """Near-ATM result is close to the exact ATM result."""
        F, T = 100.0, 1.0
        alpha, rho, nu = 0.3, -0.3, 0.4
        sigma_atm = sabr_normal_vol(F, K=F, T=T, alpha=alpha, rho=rho, nu=nu)["sigma_normal"]
        sigma_near = sabr_normal_vol(F, K=F + 1e-8, T=T, alpha=alpha, rho=rho, nu=nu)["sigma_normal"]
        assert abs(sigma_atm - sigma_near) < 1e-5

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            sabr_normal_vol(F=100.0, K=100.0, T=1.0, alpha=0.0, rho=0.0, nu=0.4)


# ---------------------------------------------------------------------------
# TestSABRCalibration
# ---------------------------------------------------------------------------

class TestSABRCalibration:
    # True parameters for synthetic smile
    _F = 100.0
    _T = 1.0
    _alpha_true = 0.25
    _beta_true = 0.5
    _rho_true = -0.3
    _nu_true = 0.4
    _strikes = np.array([80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0])

    @classmethod
    def _synthetic_vols(cls) -> np.ndarray:
        return _make_smile(
            cls._F, cls._T, cls._strikes,
            cls._alpha_true, cls._beta_true, cls._rho_true, cls._nu_true,
        )

    def test_calibration_returns_dict(self):
        """Result is a dict with all required keys."""
        market_vols = self._synthetic_vols()
        result = calibrate_sabr(self._F, self._T, self._strikes, market_vols, beta=self._beta_true)
        required_keys = {"alpha", "beta", "rho", "nu", "rmse", "fitted_vols", "market_vols", "strikes"}
        assert required_keys.issubset(result.keys())

    def test_calibration_recovers_params(self):
        """Recovered parameters are close to the true synthetic parameters."""
        market_vols = self._synthetic_vols()
        result = calibrate_sabr(self._F, self._T, self._strikes, market_vols, beta=self._beta_true)
        assert abs(result["alpha"] - self._alpha_true) < 0.05, (
            f"alpha mismatch: got {result['alpha']:.4f}, expected ~{self._alpha_true}"
        )
        assert abs(result["rho"] - self._rho_true) < 0.1, (
            f"rho mismatch: got {result['rho']:.4f}, expected ~{self._rho_true}"
        )
        assert abs(result["nu"] - self._nu_true) < 0.15, (
            f"nu mismatch: got {result['nu']:.4f}, expected ~{self._nu_true}"
        )

    def test_fit_quality(self):
        """RMSE on synthetic data (true params known) must be < 0.005."""
        market_vols = self._synthetic_vols()
        result = calibrate_sabr(self._F, self._T, self._strikes, market_vols, beta=self._beta_true)
        assert result["rmse"] < 0.005, f"RMSE too large: {result['rmse']:.6f}"

    def test_fitted_vols_shape(self):
        """fitted_vols is an ndarray of the same length as strikes."""
        market_vols = self._synthetic_vols()
        result = calibrate_sabr(self._F, self._T, self._strikes, market_vols, beta=self._beta_true)
        assert isinstance(result["fitted_vols"], np.ndarray)
        assert len(result["fitted_vols"]) == len(self._strikes)

    def test_beta_is_preserved(self):
        """Calibration keeps the fixed beta exactly."""
        market_vols = self._synthetic_vols()
        result = calibrate_sabr(self._F, self._T, self._strikes, market_vols, beta=0.7)
        assert result["beta"] == 0.7

    def test_with_initial_guess(self):
        """Calibration accepts an initial_guess dict and converges."""
        market_vols = self._synthetic_vols()
        result = calibrate_sabr(
            self._F, self._T, self._strikes, market_vols,
            beta=self._beta_true,
            initial_guess={"rho": -0.2, "nu": 0.35},
        )
        assert result["rmse"] < 0.005

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            calibrate_sabr(100.0, 1.0, np.array([90.0, 100.0]), np.array([0.3]))
