"""Thin workflow wrappers for the v0.2 quant-engine modules.

These follow the same BaseWorkflow pattern as pricing_workflow.py but wrap the
new high-precision engines: Fourier, American LSM, Merton, Heston ADI PDE,
SABR, and Dupire local-vol.

CLI commands call engines directly for simplicity; these wrappers exist for
programmatic use and for any future workflow-level orchestration.
"""
from __future__ import annotations

import numpy as np

from workflows.base import BaseWorkflow


class QuantPricingWorkflow(BaseWorkflow):
    """Orchestrates pricing with the v0.2 quant engines."""

    # ---------------------------------------------------------------- Fourier

    def run_fourier(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        v0: float,
        is_call: bool = True,
        method: str = "cos",
    ) -> dict:
        """Price a Heston option via Fourier methods.

        method : {"cos", "carr-madan", "lewis", "all"}
        Returns a dict keyed by method name(s).
        """
        from engines.pricing.heston_fourier import (
            heston_price_carr_madan,
            heston_price_cos,
            heston_price_lewis,
        )

        results = {}
        pkw = dict(S0=S0, K=K, T=T, r=r, kappa=kappa, theta=theta,
                   sigma_v=sigma_v, rho=rho, v0=v0, is_call=is_call)
        if method in ("cos", "all"):
            results["cos"] = heston_price_cos(**pkw)  # type: ignore[arg-type]
        if method in ("carr-madan", "all"):
            results["carr-madan"] = heston_price_carr_madan(**pkw)  # type: ignore[arg-type]
        if method in ("lewis", "all"):
            results["lewis"] = heston_price_lewis(**pkw)  # type: ignore[arg-type]
        return results

    # ---------------------------------------------------------------- American LSM

    def run_american(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        is_call: bool = False,
        n_paths: int = 20_000,
        n_steps: int = 100,
        poly_degree: int = 3,
        seed: int | None = None,
    ) -> dict:
        """Price an American option via Longstaff-Schwartz LSM."""
        from engines.pricing.american_mc import american_option_lsm
        return american_option_lsm(
            S0=S0, K=K, T=T, r=r, sigma=sigma,
            is_call=is_call, n_paths=n_paths, n_steps=n_steps,
            poly_degree=poly_degree, seed=seed,
        )

    # ---------------------------------------------------------------- Merton

    def run_merton(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        lam: float,
        mu_J: float,
        sigma_J: float,
        is_call: bool = True,
        n_terms: int = 50,
    ) -> dict:
        """Price a European option under Merton (1976) jump-diffusion."""
        from engines.pricing.jump_diffusion import merton_price
        return merton_price(
            S0=S0, K=K, T=T, r=r, sigma=sigma,
            lam=lam, mu_J=mu_J, sigma_J=sigma_J,
            is_call=is_call, n_terms=n_terms,
        )

    # ---------------------------------------------------------------- Heston ADI PDE

    def run_heston_pde(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        v0: float,
        is_call: bool = True,
        N_x: int = 80,
        N_v: int = 40,
        N_t: int = 80,
    ) -> dict:
        """Price a Heston European option via 2-D Douglas-Rachford ADI PDE."""
        from engines.pricing.pde import heston_pde_price
        return heston_pde_price(
            S0=S0, K=K, T=T, r=r,
            kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho, v0=v0,
            is_call=is_call, N_x=N_x, N_v=N_v, N_t=N_t,
        )

    # ---------------------------------------------------------------- SABR

    def run_sabr_vol(
        self,
        F: float,
        K: float,
        T: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        correction: str = "hagan",
    ) -> dict:
        """SABR (Hagan 2002) lognormal implied vol."""
        from engines.pricing.sabr import sabr_implied_vol
        return sabr_implied_vol(F=F, K=K, T=T, alpha=alpha, beta=beta,
                                rho=rho, nu=nu, correction=correction)

    def run_sabr_calibrate(
        self,
        F: float,
        T: float,
        strikes: np.ndarray,
        market_vols: np.ndarray,
        beta: float = 0.5,
    ) -> dict:
        """Calibrate SABR (α, ρ, ν) to a market smile."""
        from engines.pricing.sabr import calibrate_sabr
        return calibrate_sabr(F=F, T=T, strikes=np.asarray(strikes),
                              market_vols=np.asarray(market_vols), beta=beta)

    # ---------------------------------------------------------------- Local vol / SVI

    def run_dupire(
        self,
        K: float,
        T: float,
        S0: float,
        r: float,
        iv_surface: object,
    ) -> dict:
        """Dupire local vol at (K, T) from a callable implied-vol surface."""
        from engines.pricing.local_vol import dupire_local_vol
        return dupire_local_vol(K=K, T=T, S0=S0, r=r, iv_surface=iv_surface)  # type: ignore[arg-type]

    def run_svi_calibrate(
        self,
        log_strikes: np.ndarray,
        market_total_var: np.ndarray,
    ) -> dict:
        """Fit Gatheral SVI raw parametrisation to a maturity slice."""
        from engines.pricing.local_vol import calibrate_svi
        return calibrate_svi(np.asarray(log_strikes), np.asarray(market_total_var))
