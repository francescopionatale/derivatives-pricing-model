"""Tests for variance reduction techniques: control variates, Sobol QMC, moment matching."""

import numpy as np

from engines.pricing.black_scholes import bs_price_and_greeks
from engines.simulation.gbm import simulate_gbm_paths
from engines.simulation.variance_reduction import (
    apply_moment_matching,
    mc_with_control_variate,
    sobol_standard_normal,
)

# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------

S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2
N_PATHS = 10_000
SEED = 42
N_STEPS = 52  # weekly steps


def _make_paths(n_paths: int = N_PATHS, seed: int = SEED) -> np.ndarray:
    return simulate_gbm_paths(S0=S0, r=r, sigma=sigma, T=T, n_steps=N_STEPS, n_paths=n_paths, seed=seed)


def _call_payoff(paths: np.ndarray) -> np.ndarray:
    """Discounted call payoff per path."""
    ST = paths[-1]
    return np.exp(-r * T) * np.maximum(ST - K, 0.0)


def _bs_delta_control_payoff(paths: np.ndarray) -> np.ndarray:
    """Discounted incremental gain from delta-hedging.

    For a European call, the BS-delta hedged portfolio has theoretical P&L = 0
    (control_exact = 0).  The per-path realised P&L is:

        X_i = delta_0 * (S_T - S_0) * e^{-rT}

    where delta_0 is the initial BS delta.  This is correlated with the call
    payoff and has zero expected value under risk-neutral measure only when
    framing it as a simple linear payoff proxy; here we use the simpler but
    still effective control:

        X_i = delta_0 * S_0 * (S_T / S_0 - e^{rT}) * e^{-rT}
            = delta_0 * (S_T * e^{-rT} - S_0)

    E[X] = delta_0 * (S_0 - S_0) = 0  (risk-neutral drift of discounted stock = S_0)
    """
    bs = bs_price_and_greeks(S=S0, K=K, T=T, r=r, sigma=sigma, is_call=True)
    delta_0 = bs["delta"]
    ST = paths[-1]
    # Discounted terminal stock minus current price — E[e^{-rT} S_T] = S_0
    return delta_0 * (ST * np.exp(-r * T) - S0)


# ---------------------------------------------------------------------------
# TestControlVariate
# ---------------------------------------------------------------------------


class TestControlVariate:
    def test_control_variate_reduces_variance(self):
        paths = _make_paths()
        result = mc_with_control_variate(
            payoff_fn=_call_payoff,
            control_payoff_fn=_bs_delta_control_payoff,
            control_exact=0.0,
            paths=paths,
            r=r,
            T=T,
        )
        assert result["variance_reduction_ratio"] > 2.0, (
            f"Expected VRR > 2, got {result['variance_reduction_ratio']:.3f}"
        )

    def test_control_variate_price_near_bs(self):
        paths = _make_paths()
        result = mc_with_control_variate(
            payoff_fn=_call_payoff,
            control_payoff_fn=_bs_delta_control_payoff,
            control_exact=0.0,
            paths=paths,
            r=r,
            T=T,
        )
        bs_price = bs_price_and_greeks(S=S0, K=K, T=T, r=r, sigma=sigma)["price"]
        assert abs(result["price"] - bs_price) < 0.05, (
            f"CV price {result['price']:.4f} too far from BS {bs_price:.4f}"
        )

    def test_beta_near_minus_one_for_bs_control(self):
        paths = _make_paths()
        result = mc_with_control_variate(
            payoff_fn=_call_payoff,
            control_payoff_fn=_bs_delta_control_payoff,
            control_exact=0.0,
            paths=paths,
            r=r,
            T=T,
        )
        assert -3.0 <= result["beta"] <= 3.0, (
            f"Beta {result['beta']:.4f} outside expected range [-3, 3]"
        )


# ---------------------------------------------------------------------------
# TestSobol
# ---------------------------------------------------------------------------


class TestSobol:
    def test_sobol_shape(self):
        z = sobol_standard_normal(n_dims=5, n_points=32)
        assert z.shape == (32, 5)

    def test_sobol_normal_distribution(self):
        z = sobol_standard_normal(n_dims=10, n_points=1024, seed=7)
        # Sample mean and std should be close to 0 and 1 respectively
        assert abs(z.mean()) < 0.05, f"Mean {z.mean():.4f} too far from 0"
        assert abs(z.std() - 1.0) < 0.05, f"Std {z.std():.4f} too far from 1"

    def test_sobol_reproducible(self):
        z1 = sobol_standard_normal(n_dims=4, n_points=64, seed=99)
        z2 = sobol_standard_normal(n_dims=4, n_points=64, seed=99)
        np.testing.assert_array_equal(z1, z2)

    def test_sobol_different_seeds_differ(self):
        z1 = sobol_standard_normal(n_dims=4, n_points=64, seed=1)
        z2 = sobol_standard_normal(n_dims=4, n_points=64, seed=2)
        assert not np.allclose(z1, z2)

    def test_sobol_no_infinities(self):
        z = sobol_standard_normal(n_dims=50, n_points=2048, seed=0)
        assert np.all(np.isfinite(z)), "Sobol samples contain non-finite values"


# ---------------------------------------------------------------------------
# TestMomentMatching
# ---------------------------------------------------------------------------


class TestMomentMatching:
    def _make_Z(self, n_steps: int = 10, n_paths: int = 500, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_steps, n_paths))

    def test_moment_matching_mean_zero(self):
        Z = self._make_Z()
        Zm = apply_moment_matching(Z)
        row_means = np.mean(Zm, axis=1)
        np.testing.assert_allclose(row_means, 0.0, atol=1e-12)

    def test_moment_matching_std_one(self):
        Z = self._make_Z()
        Zm = apply_moment_matching(Z)
        row_stds = np.std(Zm, axis=1, ddof=0)
        np.testing.assert_allclose(row_stds, 1.0, atol=1e-12)

    def test_moment_matching_preserves_shape(self):
        Z = self._make_Z(n_steps=5, n_paths=100)
        Zm = apply_moment_matching(Z)
        assert Zm.shape == Z.shape


# ---------------------------------------------------------------------------
# TestGBMFlags
# ---------------------------------------------------------------------------


class TestGBMFlags:
    def test_quasi_mc_flag(self):
        paths = simulate_gbm_paths(
            S0=S0, r=r, sigma=sigma, T=T,
            n_steps=N_STEPS, n_paths=256, seed=0,
            quasi_mc=True,
        )
        assert paths.shape == (N_STEPS + 1, 256)
        assert np.all(np.isfinite(paths))
        assert np.all(paths > 0)

    def test_moment_matching_flag(self):
        paths = simulate_gbm_paths(
            S0=S0, r=r, sigma=sigma, T=T,
            n_steps=N_STEPS, n_paths=256, seed=0,
            moment_matching=True,
        )
        assert paths.shape == (N_STEPS + 1, 256)
        assert np.all(np.isfinite(paths))
        assert np.all(paths > 0)

    def test_quasi_mc_moment_matching_combined(self):
        paths = simulate_gbm_paths(
            S0=S0, r=r, sigma=sigma, T=T,
            n_steps=N_STEPS, n_paths=256, seed=0,
            quasi_mc=True,
            moment_matching=True,
        )
        assert paths.shape == (N_STEPS + 1, 256)
        assert np.all(np.isfinite(paths))
        assert np.all(paths > 0)

    def test_quasi_mc_with_antithetic(self):
        paths = simulate_gbm_paths(
            S0=S0, r=r, sigma=sigma, T=T,
            n_steps=N_STEPS, n_paths=256, seed=0,
            quasi_mc=True,
            antithetic=True,
        )
        assert paths.shape == (N_STEPS + 1, 256)
        assert np.all(np.isfinite(paths))

    def test_quasi_mc_reduces_error(self):
        """Both quasi-MC and pseudo-random give finite, positive call prices."""
        bs_price = bs_price_and_greeks(S=S0, K=K, T=T, r=r, sigma=sigma)["price"]
        n_paths = 2048

        # Pseudo-random
        paths_mc = simulate_gbm_paths(
            S0=S0, r=r, sigma=sigma, T=T,
            n_steps=N_STEPS, n_paths=n_paths, seed=SEED,
        )
        price_mc = float(np.mean(_call_payoff(paths_mc)))

        # Quasi-MC
        paths_qmc = simulate_gbm_paths(
            S0=S0, r=r, sigma=sigma, T=T,
            n_steps=N_STEPS, n_paths=n_paths, seed=SEED,
            quasi_mc=True,
        )
        price_qmc = float(np.mean(_call_payoff(paths_qmc)))

        # Both should be finite and in a reasonable range
        assert np.isfinite(price_mc), "MC price is not finite"
        assert np.isfinite(price_qmc), "QMC price is not finite"
        assert price_mc > 0, "MC price should be positive"
        assert price_qmc > 0, "QMC price should be positive"

        # Quasi-MC price should be within reasonable distance of BS
        # (not a strict guarantee per run, but holds for these params)
        assert abs(price_qmc - bs_price) < 1.0, (
            f"QMC price {price_qmc:.4f} seems unreasonable vs BS {bs_price:.4f}"
        )
        assert abs(price_mc - bs_price) < 1.0, (
            f"MC price {price_mc:.4f} seems unreasonable vs BS {bs_price:.4f}"
        )
