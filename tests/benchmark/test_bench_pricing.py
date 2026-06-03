"""Benchmarks for core pricing engines via pytest-benchmark.

Run with: pytest tests/benchmark/ --benchmark-only -v
Or:        pytest tests/benchmark/ --benchmark-sort=mean
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def heston_params():
    return dict(S0=100.0, K=100.0, T=1.0, r=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04)


@pytest.fixture
def bs_params():
    return dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


# ---------------------------------------------------------------------------
# Black-Scholes analytical
# ---------------------------------------------------------------------------

class TestBSBenchmark:
    def test_bs_pricing(self, benchmark, bs_params):
        from engines.pricing.black_scholes import bs_price_and_greeks
        result = benchmark(bs_price_and_greeks, **bs_params)
        assert result["price"] > 0


# ---------------------------------------------------------------------------
# Fourier methods
# ---------------------------------------------------------------------------

class TestFourierBenchmark:
    def test_cos_atm(self, benchmark, heston_params):
        from engines.pricing.heston_fourier import heston_price_cos
        result = benchmark(heston_price_cos, **heston_params)
        assert result["price"] > 0

    def test_cos_array_strikes(self, benchmark, heston_params):
        from engines.pricing.heston_fourier import heston_price_cos
        strikes = np.array([80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0])

        def price_strip():
            return heston_price_cos(**{**heston_params, "K": strikes})

        result = benchmark(price_strip)
        assert result["price"].shape == (9,)

    def test_carr_madan_strip(self, benchmark, heston_params):
        from engines.pricing.heston_fourier import heston_price_carr_madan
        result = benchmark(heston_price_carr_madan, **heston_params)
        assert result["price"] > 0


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

class TestMCBenchmark:
    def test_gbm_10k_paths(self, benchmark):
        from engines.simulation.gbm import simulate_gbm_paths

        def run():
            return simulate_gbm_paths(S0=100, r=0.05, sigma=0.2, T=1, n_steps=252, n_paths=10_000, seed=42)

        paths = benchmark(run)
        assert paths.shape == (253, 10_000)

    def test_heston_qe_10k_paths(self, benchmark, heston_params):
        from engines.simulation.heston import simulate_heston_paths_qe

        def run():
            return simulate_heston_paths_qe(
                S0=heston_params["S0"], v0=heston_params["v0"],
                kappa=heston_params["kappa"], theta=heston_params["theta"],
                sigma_v=heston_params["sigma_v"], rho=heston_params["rho"],
                r=heston_params["r"], T=heston_params["T"],
                n_steps=252, n_paths=10_000, seed=42,
            )

        S, V = benchmark(run)
        assert S.shape == (253, 10_000)


# ---------------------------------------------------------------------------
# PDE
# ---------------------------------------------------------------------------

class TestPDEBenchmark:
    def test_bs_pde_cn(self, benchmark, bs_params):
        from engines.pricing.pde import bs_pde_price

        def run():
            return bs_pde_price(S0=bs_params["S"], K=bs_params["K"], T=bs_params["T"],
                                 r=bs_params["r"], sigma=bs_params["sigma"],
                                 N_x=200, N_t=100)

        result = benchmark(run)
        assert result["price"] > 0

    def test_heston_pde_adi(self, benchmark, heston_params):
        from engines.pricing.pde import heston_pde_price

        def run():
            return heston_pde_price(**heston_params, N_x=60, N_v=30, N_t=60)

        result = benchmark(run)
        assert result["price"] > 0
