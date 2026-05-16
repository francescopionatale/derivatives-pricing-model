"""
Generates output/visualizations.pdf — a multi-page PDF containing every plot in the
visualization module, using synthetic seeded data.

Usage:
    PYTHONPATH=src python scripts/generate_visualizations_report.py
"""
import sys
from pathlib import Path

# Allow running from the repo root without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from derivatives_pricing_model.visualization import theme
from derivatives_pricing_model.visualization.plots import (
    plot_greek_curves,
    plot_greek_bar,
    plot_payoff_diagram,
    plot_mc_convergence,
    plot_pnl_distribution,
    plot_pnl_vs_final_spot,
    plot_pnl_surface,
    plot_pnl_3d_surface,
    plot_pnl_attribution,
    plot_hedging_paths,
    plot_pnl_comparison,
    plot_stress_scenario_comparison,
    plot_implied_vol_surface,
    plot_implied_vol_smile,
    plot_historical_volatility,
)
from derivatives_pricing_model.engines.pricing.black_scholes import bs_price_and_greeks
from derivatives_pricing_model.engines.simulation.gbm import simulate_gbm_paths
from derivatives_pricing_model.engines.stress.scenario import generate_student_t_paths
from derivatives_pricing_model.engines.hedging.discrete_hedging import simulate_discrete_hedging


def _synthetic_iv_surface(S0: float, rng: np.random.Generator):
    maturities_list = [0.25, 0.5, 1.0, 2.0]
    moneyness_levels = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
    ks, ts, ivs = [], [], []
    for T in maturities_list:
        for m in moneyness_levels:
            K = S0 * m
            iv = 0.20 + 0.06 * (m - 1.0) ** 2 - 0.025 * (m - 1.0) + rng.normal(0, 0.003)
            ivs.append(max(0.05, iv))
            ks.append(K)
            ts.append(T)
    return np.array(ks), np.array(ts), np.array(ivs)


def main():
    OUT = Path("output/visualizations.pdf")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    M, n_steps = 2000, 80

    print("Generating paths …")
    paths_base = simulate_gbm_paths(S0, r, sigma, T, n_steps, M, seed=42, antithetic=True)
    paths_stress = generate_student_t_paths(S0, r, sigma, T, n_steps, M, df=4.0, seed=42)

    print("Running discrete hedging (baseline, with path tracking) …")
    res_base = simulate_discrete_hedging(
        paths_base, K, T, r, sigma, is_call=True, transaction_cost=0.001, track_paths=True
    )
    print("Running discrete hedging (stress) …")
    res_stress = simulate_discrete_hedging(
        paths_stress, K, T, r, sigma, is_call=True, transaction_cost=0.001
    )

    pnls_base = res_base["total_pnl"]
    pnls_stress = res_stress["total_pnl"]
    final_spots_base = paths_base[-1, :]

    # MC convergence data
    discounted_payoffs = np.exp(-r * T) * np.maximum(paths_base[-1] - K, 0)
    n_arr = np.arange(1, len(discounted_payoffs) + 1)
    cum_sum = np.cumsum(discounted_payoffs)
    cum_sum_sq = np.cumsum(discounted_payoffs ** 2)
    cum_prices = cum_sum / n_arr
    cum_var = np.maximum(cum_sum_sq / n_arr - (cum_sum / n_arr) ** 2, 0.0)
    cum_stderr = np.sqrt(cum_var / n_arr)
    bs_ref = bs_price_and_greeks(S0, K, T, r, sigma, is_call=True)["price"]

    # Stress scenario results (synthetic)
    from derivatives_pricing_model.engines.stress.scenario import calculate_var_es, apply_spot_vol_shock, generate_short_convexity_scenario

    def _make_scenario_result(pnls):
        tail = calculate_var_es(pnls, confidence_level=0.95)
        return {"var_95": tail["var"], "es_95": tail["es"]}

    shocked_S0, shocked_sigma = apply_spot_vol_shock(S0, sigma, -0.10, 0.05)
    paths_sv = simulate_gbm_paths(shocked_S0, r, shocked_sigma, T, n_steps, M, seed=42, antithetic=True)
    res_sv = simulate_discrete_hedging(paths_sv, K, T, r, shocked_sigma, is_call=True)

    paths_sc = generate_short_convexity_scenario(S0, r, sigma, T, n_steps, M, seed=42)
    res_sc = simulate_discrete_hedging(paths_sc, K, T, r, sigma, is_call=True)

    stress_results = {
        "gaussian": _make_scenario_result(pnls_base),
        "student_t": _make_scenario_result(pnls_stress),
        "spot_vol_shock": _make_scenario_result(res_sv["total_pnl"]),
        "short_convexity": _make_scenario_result(res_sc["total_pnl"]),
    }

    # Implied vol surface data
    iv_strikes, iv_mats, iv_vols = _synthetic_iv_surface(S0, rng)

    # Historical vol synthetic data
    n_days = 500
    hist_prices = S0 * np.exp(np.cumsum(rng.normal(0, sigma / np.sqrt(252), n_days)))
    hist_dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    log_rets = np.diff(np.log(hist_prices))
    window = 21
    hist_vols = np.array(
        [np.std(log_rets[max(0, i - window):i]) * np.sqrt(252) for i in range(1, n_days)]
    )
    hist_vols_padded = np.full(n_days, np.nan)
    hist_vols_padded[window:] = hist_vols[window - 1:]
    valid = ~np.isnan(hist_vols_padded)
    hist_dates_valid = hist_dates[valid]
    hist_vols_valid = hist_vols_padded[valid]
    hist_prices_valid = hist_prices[valid]

    print(f"Writing {OUT} …")
    with PdfPages(OUT) as pdf:
        theme.PDF_OUTPUT = pdf

        # ── 1. Greek sensitivity curves ──────────────────────────────────────
        plot_greek_curves(S0, K, T, r, sigma, is_call=True)

        # ── 2. Greek bar chart (backward-compat view) ────────────────────────
        plot_greek_bar(bs_price_and_greeks(S0, K, T, r, sigma, is_call=True))

        # ── 3. Payoff diagrams (call and put) ────────────────────────────────
        premium_call = bs_ref
        premium_put = bs_price_and_greeks(S0, K, T, r, sigma, is_call=False)["price"]
        plot_payoff_diagram(S0, K, premium_call, is_call=True)
        plot_payoff_diagram(S0, K, premium_put, is_call=False)

        # ── 4. MC convergence ────────────────────────────────────────────────
        plot_mc_convergence(cum_prices, cum_stderr, analytical_price=bs_ref)

        # ── 5. P&L distribution ──────────────────────────────────────────────
        plot_pnl_distribution(pnls_base)

        # ── 6. P&L vs final spot ─────────────────────────────────────────────
        plot_pnl_vs_final_spot(final_spots_base, pnls_base, K)

        # ── 7. P&L density (hexbin) ──────────────────────────────────────────
        plot_pnl_surface(final_spots_base, pnls_base, strike=K)

        # ── 8. 3D P&L surface ────────────────────────────────────────────────
        plot_pnl_3d_surface(final_spots_base, pnls_base)

        # ── 9. P&L attribution waterfall ─────────────────────────────────────
        plot_pnl_attribution(res_base)

        # ── 10. Hedging path fan chart ───────────────────────────────────────
        plot_hedging_paths(res_base["cumulative_pnl_paths"])

        # ── 11. Baseline vs stress comparison ────────────────────────────────
        plot_pnl_comparison(pnls_base, pnls_stress)

        # ── 12. Stress scenario comparison ───────────────────────────────────
        plot_stress_scenario_comparison(stress_results)

        # ── 13. Implied vol surface (3D) ─────────────────────────────────────
        plot_implied_vol_surface(iv_strikes, iv_mats, iv_vols)

        # ── 14. Implied vol smile (moneyness) ────────────────────────────────
        plot_implied_vol_smile(iv_strikes, iv_mats, iv_vols, S0)

        # ── 15. Historical realized volatility ───────────────────────────────
        plot_historical_volatility(hist_dates_valid, hist_vols_valid, window,
                                   prices=hist_prices_valid)

        theme.PDF_OUTPUT = None

    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"Done. {OUT}  ({size_mb:.1f} MB, 15 plots)")


if __name__ == "__main__":
    main()
