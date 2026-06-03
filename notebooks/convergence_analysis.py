#!/usr/bin/env python3
"""Convergence analysis for the quant engine upgrade.

Generates publication-quality PNG figures in docs/figures/ showing:
  1. MC 1/√N convergence envelope (GBM ATM call)
  2. Euler vs QE bias (Heston ATM call vs Fourier benchmark)
  3. Variance-reduction ratio (antithetic + QMC vs plain MC)
  4. BS PDE order of convergence (Δx → 0)
  5. Fourier convergence to machine precision (COS N terms)
  6. SABR + Heston smile fits vs implied vol surface
  7. LSM exercise boundary
  8. Binomial oscillation (convergence with steps)

Run from the repository root:
    MPLBACKEND=Agg PYTHONPATH=src python notebooks/convergence_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

from visualization.theme import PALETTE, apply_theme  # noqa: E402

apply_theme()


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 1. MC 1/√N convergence envelope
# ---------------------------------------------------------------------------
def plot_mc_convergence():
    print("1. MC convergence envelope …")
    from engines.pricing.black_scholes import bs_price_and_greeks
    from engines.simulation.gbm import simulate_gbm_paths

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    true_price = bs_price_and_greeks(S=S0, K=K, T=T, r=r, sigma=sigma)["price"]

    path_counts = [200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
    means, stds = [], []
    rng = np.random.default_rng(0)
    for n in path_counts:
        paths = simulate_gbm_paths(S0, r, sigma, T, n_steps=252, n_paths=n, seed=rng.integers(1_000_000))
        payoffs = np.maximum(paths[-1] - K, 0.0)
        disc = np.exp(-r * T)
        est = disc * payoffs.mean()
        std = disc * payoffs.std(ddof=1) / np.sqrt(n)
        means.append(est)
        stds.append(std)

    means, stds = np.array(means), np.array(stds)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(true_price, color=PALETTE["line_bs"], lw=1.5, label="BS true price")
    ax.fill_between(path_counts, means - 2 * stds, means + 2 * stds,
                    alpha=0.25, color=PALETTE["fill_ci"], label="±2σ envelope")
    ax.plot(path_counts, means, "o-", color=PALETTE["line_primary"], ms=4, label="MC estimate")
    # 1/√N reference line
    ref = true_price + stds[0] * np.sqrt(path_counts[0]) / np.sqrt(np.array(path_counts))
    ax.plot(path_counts, true_price + ref - ref, "--", color="grey", lw=1, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Number of paths $N$")
    ax.set_ylabel("Option price")
    ax.set_title("MC ATM call: convergence vs $N$")
    ax.legend(fontsize=8)
    _save(fig, "mc_convergence")


# ---------------------------------------------------------------------------
# 2. Euler vs QE bias
# ---------------------------------------------------------------------------
def plot_euler_vs_qe():
    print("2. Euler vs QE bias …")
    from engines.pricing.heston_fourier import heston_price_cos
    from engines.simulation.heston import simulate_heston_paths_euler, simulate_heston_paths_qe

    params = dict(S0=100, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.05, T=1)
    K = 100.0
    true_price = heston_price_cos(**params, K=K)["price"]

    step_counts = [16, 32, 64, 128, 256]

    def mc_price(sim_fn, n_steps, seed=42):
        S, _ = sim_fn(**params, n_steps=n_steps, n_paths=20_000, seed=seed)
        payoffs = np.maximum(S[-1] - K, 0.0)
        return float(np.exp(-params["r"] * params["T"]) * payoffs.mean())

    euler_prices = [mc_price(simulate_heston_paths_euler, n) for n in step_counts]
    qe_prices = [mc_price(simulate_heston_paths_qe, n) for n in step_counts]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(true_price, color=PALETTE["line_bs"], lw=1.5, label="Fourier (COS) reference")
    ax.plot(step_counts, euler_prices, "s--", color="#d4290a", ms=5, label="Euler-Maruyama")
    ax.plot(step_counts, qe_prices, "o-", color=PALETTE["line_primary"], ms=5, label="QE scheme")
    ax.set_xlabel("Time steps $N_t$")
    ax.set_ylabel("Option price")
    ax.set_title("Heston ATM call: Euler vs QE discretisation bias")
    ax.legend(fontsize=8)
    _save(fig, "euler_vs_qe")


# ---------------------------------------------------------------------------
# 3. Variance reduction ratios
# ---------------------------------------------------------------------------
def plot_variance_reduction():
    print("3. Variance reduction ratios …")
    from engines.simulation.gbm import simulate_gbm_paths

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    n_paths, n_steps, seed = 5_000, 252, 42

    disc = np.exp(-r * T)

    def run(paths):
        payoffs = np.maximum(paths[-1] - K, 0.0)
        return disc * payoffs, float((disc * payoffs).std(ddof=1))

    plain = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)
    anti = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed, antithetic=True)
    mm = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed, moment_matching=True)
    qmc = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed, quasi_mc=True)

    _, std_plain = run(plain)
    _, std_anti = run(anti)
    _, std_mm = run(mm)
    _, std_qmc = run(qmc)

    labels = ["Plain MC", "Antithetic", "Moment\nmatching", "Sobol QMC"]
    ratios = [1.0, (std_plain / std_anti) ** 2, (std_plain / std_mm) ** 2, (std_plain / std_qmc) ** 2]
    colors = ["#888888", PALETTE["line_primary"], PALETTE["line_mean"], PALETTE["line_var"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, ratios, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    for bar, v in zip(bars, ratios, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{v:.1f}×", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Variance reduction ratio")
    ax.set_title("Variance reduction: effective sample multiplier")
    _save(fig, "variance_reduction")


# ---------------------------------------------------------------------------
# 4. BS PDE order of convergence
# ---------------------------------------------------------------------------
def plot_pde_convergence():
    print("4. PDE order of convergence …")
    from engines.pricing.black_scholes import bs_price_and_greeks
    from engines.pricing.pde import bs_pde_price

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    true_price = bs_price_and_greeks(S=S0, K=K, T=T, r=r, sigma=sigma)["price"]

    grid_sizes = [25, 50, 100, 200, 400]
    errors = []
    for N in grid_sizes:
        p = bs_pde_price(S0=S0, K=K, T=T, r=r, sigma=sigma, N_x=N, N_t=N // 2)["price"]
        errors.append(abs(p - true_price))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(grid_sizes, errors, "o-", color=PALETTE["line_primary"], ms=5, label="CN error")
    # O(N^-2) reference line
    ref = errors[0] * (np.array(grid_sizes) / grid_sizes[0]) ** -2
    ax.loglog(grid_sizes, ref, "--", color="grey", lw=1, label=r"$O(N^{-2})$")
    ax.set_xlabel("Grid size $N_x$")
    ax.set_ylabel("Absolute price error")
    ax.set_title("PDE Crank-Nicolson: $O(\\Delta x^2)$ convergence")
    ax.legend(fontsize=8)
    _save(fig, "pde_convergence")


# ---------------------------------------------------------------------------
# 5. Fourier COS convergence
# ---------------------------------------------------------------------------
def plot_fourier_convergence():
    print("5. Fourier COS N-term convergence …")
    from engines.pricing.heston_fourier import heston_price_cos

    params = dict(S0=100, K=100, T=1, r=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04)
    ref = heston_price_cos(**params, N=4096)["price"]

    n_terms = [4, 8, 16, 32, 64, 128, 256, 512]
    errors = [abs(heston_price_cos(**params, N=n)["price"] - ref) for n in n_terms]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(n_terms, errors, "o-", color=PALETTE["line_primary"], ms=5)
    ax.set_xlabel("COS terms $N$")
    ax.set_ylabel("Absolute error vs N=4096")
    ax.set_title("Heston COS: exponential convergence in $N$")
    _save(fig, "fourier_cos_convergence")


# ---------------------------------------------------------------------------
# 6. SABR / Heston smile fits
# ---------------------------------------------------------------------------
def plot_smile_fits():
    print("6. Smile fits …")
    from engines.pricing.heston_fourier import heston_price_cos
    from engines.pricing.implied_vol import implied_volatility
    from engines.pricing.sabr import sabr_implied_vol

    S0, T, r = 100.0, 1.0, 0.05
    F = S0 * np.exp(r * T)
    strikes = np.linspace(75, 130, 30)

    # "Market" vols from Heston
    heston_p = dict(S0=S0, T=T, r=r, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04)
    mkt_ivs = []
    for K in strikes:
        price = heston_price_cos(**heston_p, K=float(K))["price"]
        try:
            iv = implied_volatility(price, S0, K, T, r, is_call=True)
        except Exception:
            iv = float("nan")
        mkt_ivs.append(iv)
    mkt_ivs = np.array(mkt_ivs)

    # SABR fit (alpha tuned to ATM)
    atm_iv = float(mkt_ivs[len(mkt_ivs) // 2])
    sabr_vols = np.array([
        sabr_implied_vol(F=F, K=K, T=T, alpha=atm_iv, beta=0.5, rho=-0.3, nu=0.5)["sigma"]
        for K in strikes
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(strikes, mkt_ivs * 100, "o", ms=4, color=PALETTE["line_primary"], label="Heston (market)")
    ax.plot(strikes, sabr_vols * 100, "--", color=PALETTE["line_mean"], lw=1.5, label="SABR approximation")
    ax.set_xlabel("Strike $K$")
    ax.set_ylabel("Implied vol (%)")
    ax.set_title("Heston vs SABR implied vol smile")
    ax.legend(fontsize=8)
    _save(fig, "smile_fits")


# ---------------------------------------------------------------------------
# 7. LSM exercise boundary
# ---------------------------------------------------------------------------
def plot_lsm_exercise_boundary():
    print("7. LSM exercise boundary …")
    from engines.pricing.american_mc import american_option_lsm

    res = american_option_lsm(S0=100, K=100, T=1, r=0.05, sigma=0.2, is_call=False,
                               n_paths=20_000, n_steps=100, seed=42)
    boundary = res["exercise_boundary"]
    t_grid = np.linspace(0, 1, len(boundary))
    valid = ~np.isnan(boundary)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_grid[valid], boundary[valid], "o-", ms=3, color=PALETTE["line_primary"],
            lw=1, label="Exercise boundary")
    ax.axhline(100, color=PALETTE["line_strike"], lw=1, ls="--", label="Strike K=100")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Critical spot price $S^*$")
    ax.set_title("LSM American put: early-exercise boundary")
    ax.legend(fontsize=8)
    _save(fig, "lsm_exercise_boundary")


# ---------------------------------------------------------------------------
# 8. Heston PDE vs Fourier smile cross-check
# ---------------------------------------------------------------------------
def plot_pde_vs_fourier():
    print("8. Heston PDE vs Fourier smile …")
    from engines.pricing.heston_fourier import heston_price_cos
    from engines.pricing.pde import heston_pde_price

    heston_p = dict(S0=100, T=1, r=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04)
    strikes = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]

    cos_prices = [heston_price_cos(**heston_p, K=K)["price"] for K in strikes]
    pde_prices = [heston_pde_price(**heston_p, K=K, N_x=100, N_v=50, N_t=100)["price"] for K in strikes]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax1, ax2 = axes

    ax1.plot(strikes, cos_prices, "o-", color=PALETTE["line_bs"], ms=5, label="COS (Fourier)")
    ax1.plot(strikes, pde_prices, "s--", color=PALETTE["line_primary"], ms=5, label="ADI PDE")
    ax1.set_xlabel("Strike $K$")
    ax1.set_ylabel("Call price")
    ax1.set_title("Heston call: Fourier vs 2-D ADI PDE")
    ax1.legend(fontsize=8)

    errors = [abs(p - c) for p, c in zip(pde_prices, cos_prices, strict=False)]
    ax2.bar(strikes, errors, color=PALETTE["line_primary"], edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Strike $K$")
    ax2.set_ylabel("Absolute error")
    ax2.set_title("PDE vs COS error")

    fig.tight_layout()
    _save(fig, "pde_vs_fourier")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Generating convergence figures → {FIGURES_DIR}")
    plot_mc_convergence()
    plot_euler_vs_qe()
    plot_variance_reduction()
    plot_pde_convergence()
    plot_fourier_convergence()
    plot_smile_fits()
    plot_lsm_exercise_boundary()
    plot_pde_vs_fourier()
    print(f"\nDone. {len(list(FIGURES_DIR.glob('*.png')))} figures in {FIGURES_DIR}")
