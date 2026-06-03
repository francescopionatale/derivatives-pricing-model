import numpy as np
import pytest

from engines.pricing.heston_vanilla import heston_vanilla_price_mc
from engines.simulation.heston import (
    simulate_heston_paths,
    simulate_heston_paths_qe,
)

# ---------------------------------------------------------------------------
# Shared Heston parameters used across tests
# ---------------------------------------------------------------------------
HESTON_PARAMS = dict(
    S0=100.0,
    v0=0.04,
    kappa=2.0,
    theta=0.04,
    sigma_v=0.3,
    rho=-0.7,
    r=0.05,
    T=1.0,
    n_steps=50,
    n_paths=1000,
    seed=42,
)


class TestQEScheme:
    def test_qe_paths_shape(self) -> None:
        """simulate_heston_paths_qe returns (n_steps+1, n_paths) for both S and V."""
        n_steps = HESTON_PARAMS["n_steps"]
        n_paths = HESTON_PARAMS["n_paths"]
        S, V = simulate_heston_paths_qe(**HESTON_PARAMS)
        assert S.shape == (n_steps + 1, n_paths)
        assert V.shape == (n_steps + 1, n_paths)

    def test_qe_v_non_negative(self) -> None:
        """Variance process never goes negative (guaranteed by construction)."""
        S, V = simulate_heston_paths_qe(**HESTON_PARAMS)
        assert np.all(V >= 0.0), "Variance process produced negative values"

    def test_qe_antithetic(self) -> None:
        """With antithetic=True, n_paths must be even; shapes are correct."""
        params = {**HESTON_PARAMS, "antithetic": True}  # n_paths=1000 already even
        S, V = simulate_heston_paths_qe(**params)
        n_steps = params["n_steps"]
        n_paths = params["n_paths"]
        assert S.shape == (n_steps + 1, n_paths)
        assert V.shape == (n_steps + 1, n_paths)
        assert np.all(V >= 0.0)

    def test_qe_vs_euler_convergence(self) -> None:
        """QE with 50 steps and Euler with 500 steps price within 5% of each other."""
        common = dict(
            S0=100.0,
            K=100.0,
            T=1.0,
            r=0.05,
            is_call=True,
            kappa=2.0,
            theta=0.04,
            sigma_v=0.3,
            rho=-0.7,
            v0=0.04,
            n_paths=20000,
            seed=42,
        )
        qe_result = heston_vanilla_price_mc(**common, n_steps=50, scheme="qe")
        euler_result = heston_vanilla_price_mc(**common, n_steps=500, scheme="euler")

        qe_price = qe_result["price"]
        euler_price = euler_result["price"]
        ref_price = max(abs(euler_price), 1e-8)
        rel_diff = abs(qe_price - euler_price) / ref_price

        assert rel_diff < 0.05, (
            f"QE (50 steps) price {qe_price:.4f} differs from "
            f"Euler (500 steps) price {euler_price:.4f} by {rel_diff:.2%} (> 5%)"
        )

    def test_scheme_dispatcher(self) -> None:
        """simulate_heston_paths with scheme='euler' and scheme='qe' both return correct shapes."""
        n_steps = HESTON_PARAMS["n_steps"]
        n_paths = HESTON_PARAMS["n_paths"]

        S_euler, V_euler = simulate_heston_paths(**HESTON_PARAMS, scheme="euler")
        assert S_euler.shape == (n_steps + 1, n_paths)
        assert V_euler.shape == (n_steps + 1, n_paths)

        S_qe, V_qe = simulate_heston_paths(**HESTON_PARAMS, scheme="qe")
        assert S_qe.shape == (n_steps + 1, n_paths)
        assert V_qe.shape == (n_steps + 1, n_paths)

    def test_scheme_dispatcher_invalid(self) -> None:
        """simulate_heston_paths raises ValueError for unknown scheme."""
        with pytest.raises(ValueError, match="Unknown scheme"):
            simulate_heston_paths(**HESTON_PARAMS, scheme="bogus")

    def test_heston_vanilla_mc_accepts_scheme(self) -> None:
        """heston_vanilla_price_mc accepts both scheme='qe' and scheme='euler' without error."""
        common = dict(
            S0=100.0,
            K=100.0,
            T=1.0,
            r=0.05,
            is_call=True,
            n_steps=20,
            n_paths=500,
            kappa=2.0,
            theta=0.04,
            sigma_v=0.3,
            rho=-0.7,
            v0=0.04,
            seed=0,
        )
        res_qe = heston_vanilla_price_mc(**common, scheme="qe")
        assert res_qe["price"] > 0.0

        res_euler = heston_vanilla_price_mc(**common, scheme="euler")
        assert res_euler["price"] > 0.0

    def test_qe_feller_violated(self) -> None:
        """QE is numerically stable even when Feller condition is violated (sigma_v^2 > 2*kappa*theta)."""
        # 2*kappa*theta = 2*0.5*0.01 = 0.01 < sigma_v^2 = 0.64 => Feller violated
        params = dict(
            S0=100.0,
            v0=0.04,
            kappa=0.5,
            theta=0.01,
            sigma_v=0.8,
            rho=-0.5,
            r=0.05,
            T=1.0,
            n_steps=50,
            n_paths=2000,
            seed=7,
        )
        S, V = simulate_heston_paths_qe(**params)
        assert S.shape == (params["n_steps"] + 1, params["n_paths"])
        assert V.shape == (params["n_steps"] + 1, params["n_paths"])
        assert np.all(V >= 0.0), "Variance went negative with Feller-violated params"
        assert np.all(np.isfinite(S)), "Non-finite spot prices with Feller-violated params"
        assert np.all(np.isfinite(V)), "Non-finite variances with Feller-violated params"
