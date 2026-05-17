"""Visualization plots for the derivatives pricing model."""
import numpy as np
import matplotlib.pyplot as plt

from engines.stress.scenario import calculate_var_es
from engines.pricing.black_scholes import bs_price_and_greeks
from visualization.theme import (
    apply_theme,
    _finalize,
    _strip_spines,
    PALETTE,
    FIGSIZE,
)


def plot_pnl_distribution(pnls: np.ndarray) -> None:
    from scipy.stats import gaussian_kde

    apply_theme()
    tail = calculate_var_es(pnls, confidence_level=0.95)
    var_val = -tail["var"]
    es_val = -tail["es"]

    fig, ax = plt.subplots(figsize=FIGSIZE["2d"])
    ax.hist(pnls, bins=50, density=True, alpha=0.55, color=PALETTE["line_primary"], edgecolor="none")

    x_range = np.linspace(pnls.min(), pnls.max(), 300)
    kde = gaussian_kde(pnls)
    ax.plot(x_range, kde(x_range), color=PALETTE["line_primary"], linewidth=2)

    x_tail = x_range[x_range <= var_val]
    if len(x_tail) > 1:
        ax.fill_between(x_tail, kde(x_tail), 0, alpha=0.4, color=PALETTE["fill_tail"])

    ax.axvline(var_val, color=PALETTE["line_var"], linestyle="--", linewidth=1.5,
               label=f"VaR 95%: {var_val:.3f}")
    ax.axvline(float(np.mean(pnls)), color=PALETTE["line_mean"], linestyle="--", linewidth=1.5,
               label=f"Mean: {float(np.mean(pnls)):.3f}")

    stats_text = (
        f"Mean:    {float(np.mean(pnls)):.3f}\n"
        f"Std:     {float(np.std(pnls)):.3f}\n"
        f"VaR 95%: {var_val:.3f}\n"
        f"ES 95%:  {es_val:.3f}"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            fontsize=9, family="monospace")

    ax.set_title("P&L Distribution")
    ax.set_xlabel("P&L")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    _strip_spines(ax)
    _finalize(fig, "pnl_distribution.png")


def plot_pnl_surface(final_spots: np.ndarray, pnls: np.ndarray, strike: float | None = None) -> None:
    apply_theme()
    fig, ax = plt.subplots(figsize=FIGSIZE["2d"])

    hb = ax.hexbin(final_spots, pnls, gridsize=40, cmap=PALETTE["cmap_heatmap"], mincnt=1)
    fig.colorbar(hb, ax=ax, label="Count")

    counts, xedges, yedges = np.histogram2d(final_spots, pnls, bins=40)
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
    ax.contour(X, Y, counts, levels=5, colors="white", alpha=0.4, linewidths=0.8)

    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    if strike is not None:
        ax.axvline(strike, color=PALETTE["line_strike"], linewidth=1.2, linestyle="--",
                   alpha=0.7, label=f"Strike K={strike:.0f}")
        ax.legend()

    ax.set_title("P&L vs Final Spot — Density")
    ax.set_xlabel("Final Spot Price")
    ax.set_ylabel("P&L")
    _strip_spines(ax)
    _finalize(fig, "pnl_surface.png")


def plot_pnl_3d_surface(final_spots: np.ndarray, pnls: np.ndarray) -> None:
    from scipy.ndimage import gaussian_filter

    apply_theme()
    fig = plt.figure(figsize=FIGSIZE["3d"])
    ax = fig.add_subplot(111, projection="3d")

    hist, xedges, yedges = np.histogram2d(final_spots, pnls, bins=50)
    hist_smooth = gaussian_filter(hist, sigma=1.5)

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")

    surf = ax.plot_surface(X, Y, hist_smooth, cmap=PALETTE["cmap_surface"], edgecolor="none", alpha=0.85)
    ax.plot_surface(X, Y, np.zeros_like(hist_smooth), alpha=0.1, color="gray")
    cbar_ax = fig.add_axes([0.88, 0.2, 0.025, 0.6])
    fig.colorbar(surf, cax=cbar_ax, label="Frequency")

    ax.set_title("3D P&L vs Final Spot Surface")
    ax.set_xlabel("Final Spot Price")
    ax.set_ylabel("P&L")
    ax.set_zlabel("Frequency")
    ax.view_init(elev=30, azim=-60)
    _finalize(fig, "pnl_3d_surface.png")


def plot_implied_vol_surface(
    strikes: np.ndarray, maturities: np.ndarray, implied_vols: np.ndarray
) -> None:
    from scipy.interpolate import griddata

    apply_theme()
    fig = plt.figure(figsize=FIGSIZE["3d"])
    ax = fig.add_subplot(111, projection="3d")

    k_min, k_max = strikes.min(), strikes.max()
    t_min, t_max = maturities.min(), maturities.max()
    grid_k, grid_t = np.mgrid[k_min:k_max:50j, t_min:t_max:50j]

    grid_iv = griddata((strikes, maturities), implied_vols, (grid_k, grid_t), method="cubic")
    nan_mask = np.isnan(grid_iv)
    if nan_mask.any():
        grid_iv_nn = griddata(
            (strikes, maturities), implied_vols, (grid_k, grid_t), method="nearest"
        )
        grid_iv[nan_mask] = grid_iv_nn[nan_mask]

    surf = ax.plot_surface(
        grid_k, grid_t, grid_iv,
        cmap=PALETTE["cmap_surface"], rstride=1, cstride=1, antialiased=True, alpha=0.85, edgecolor="none",
    )
    ax.scatter(strikes, maturities, implied_vols, color="black", s=15, zorder=5)

    # Fixed-position colorbar axes avoids the tilt caused by the 3D bounding box
    cbar_ax = fig.add_axes([0.88, 0.2, 0.025, 0.6])
    fig.colorbar(surf, cax=cbar_ax, label="Implied Volatility")

    ax.set_title("Implied Volatility Surface")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (Years)")
    ax.set_zlabel("Implied Volatility")
    ax.view_init(elev=25, azim=-45)
    _finalize(fig, "implied_vol_surface.png")


def plot_implied_vol_smile(
    strikes: np.ndarray,
    maturities: np.ndarray,
    implied_vols: np.ndarray,
    S0: float,
) -> None:
    apply_theme()
    fig, ax = plt.subplots(figsize=FIGSIZE["2d"])

    import matplotlib.cm as _cm
    moneyness = strikes / S0
    unique_maturities = np.unique(maturities)
    _viridis = _cm.get_cmap("viridis")
    colors = [_viridis(i / max(len(unique_maturities) - 1, 1)) for i in range(len(unique_maturities))]

    for i, T in enumerate(unique_maturities):
        mask = maturities == T
        m = moneyness[mask]
        vols = implied_vols[mask]
        sorted_idx = np.argsort(m)
        ax.plot(
            m[sorted_idx], vols[sorted_idx],
            marker="o", linestyle="-", markersize=5,
            color=colors[i % len(colors)], label=f"T={T:.2f}",
        )

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="ATM (K/S₀=1)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.set_title("Implied Volatility Smile / Skew")
    ax.set_xlabel("Moneyness (K/S₀)")
    ax.set_ylabel("Implied Volatility")
    ax.legend(title="Maturity (Years)")
    _strip_spines(ax)
    _finalize(fig, "implied_vol_smile.png")


def plot_pnl_comparison(
    pnls_base: np.ndarray,
    pnls_stress: np.ndarray,
    labels: tuple[str, str] = ("Baseline", "Stress"),
) -> None:
    from scipy.stats import gaussian_kde

    apply_theme()
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 1.5], hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    def _panel(ax, pnls, label, color):
        tail = calculate_var_es(pnls, confidence_level=0.95)
        var_val = -tail["var"]
        es_val = -tail["es"]
        ax.hist(pnls, bins=50, density=True, alpha=0.5, color=color, edgecolor="none")
        x_range = np.linspace(pnls.min(), pnls.max(), 300)
        kde = gaussian_kde(pnls)
        ax.plot(x_range, kde(x_range), color=color, linewidth=2)
        x_tail = x_range[x_range <= var_val]
        if len(x_tail) > 1:
            ax.fill_between(x_tail, kde(x_tail), 0, alpha=0.35, color=PALETTE["fill_tail"])
        ax.axvline(var_val, color=PALETTE["line_var"], linestyle="--", linewidth=1.2)
        stats = (
            f"Mean: {float(np.mean(pnls)):.3f}  Std: {float(np.std(pnls)):.3f}\n"
            f"VaR 95%: {var_val:.3f}  ES 95%: {es_val:.3f}"
        )
        ax.text(
            0.97, 0.97, stats, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            fontsize=8, family="monospace",
        )
        ax.set_title(f"{label} P&L Distribution")
        ax.set_ylabel("Density")
        _strip_spines(ax)

    _panel(ax1, pnls_base, labels[0], PALETTE["line_primary"])
    _panel(ax2, pnls_stress, labels[1], PALETTE["fill_tail"])

    x_min = min(pnls_base.min(), pnls_stress.min())
    x_max = max(pnls_base.max(), pnls_stress.max())
    x_range = np.linspace(x_min, x_max, 300)
    kde_base = gaussian_kde(pnls_base)
    kde_stress = gaussian_kde(pnls_stress)
    delta_kde = kde_stress(x_range) - kde_base(x_range)

    ax3.fill_between(x_range, delta_kde, 0,
                     where=delta_kde > 0, alpha=0.6, color=PALETTE["fill_tail"],
                     label="Higher under stress")
    ax3.fill_between(x_range, delta_kde, 0,
                     where=delta_kde <= 0, alpha=0.6, color=PALETTE["fill_profit"],
                     label="Higher under baseline")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title(f"Density Shift ({labels[1]} − {labels[0]})")
    ax3.set_xlabel("P&L")
    ax3.set_ylabel("Δ Density")
    ax3.legend(fontsize=9)
    _strip_spines(ax3)

    _finalize(fig, "pnl_comparison.png")


def plot_historical_volatility(dates, vols, window: int, prices=None) -> None:
    apply_theme()
    vols_arr = np.asarray(vols, dtype=float)
    dates_arr = np.asarray(dates)
    high_vol_thresh = float(np.nanpercentile(vols_arr, 75))

    fig, ax1 = plt.subplots(figsize=FIGSIZE["2d"])
    ax1.plot(dates_arr, vols_arr, color=PALETTE["line_primary"], linewidth=2,
             label=f"{window}-Day Rolling Volatility")
    ax1.axhline(high_vol_thresh, color=PALETTE["fill_tail"], linestyle=":", linewidth=1,
                alpha=0.6, label=f"75th pctile ({high_vol_thresh:.3f})")

    prev, start = False, None
    for i, is_high in enumerate(vols_arr >= high_vol_thresh):
        if is_high and not prev:
            start = dates_arr[i]
        elif not is_high and prev and start is not None:
            ax1.axvspan(start, dates_arr[i - 1], alpha=0.15, color=PALETTE["fill_tail"])
            start = None
        prev = bool(is_high)
    if prev and start is not None:
        ax1.axvspan(start, dates_arr[-1], alpha=0.15, color=PALETTE["fill_tail"])

    ax1.set_title("Historical Realized Volatility")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Annualized Volatility", color=PALETTE["line_primary"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["line_primary"])
    _strip_spines(ax1)

    if prices is not None:
        prices_arr = np.asarray(prices, dtype=float)
        ax2 = ax1.twinx()
        ax2.plot(dates_arr, prices_arr, color="gray", linewidth=1, alpha=0.4, label="Price")
        ax2.set_ylabel("Price", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        ax1.legend(loc="upper left")

    plt.tight_layout()
    _finalize(fig, "historical_volatility.png")


def plot_pnl_vs_final_spot(final_spots: np.ndarray, pnls: np.ndarray, strike: float) -> None:
    apply_theme()
    fig, ax = plt.subplots(figsize=FIGSIZE["2d"])

    ax.scatter(final_spots, pnls, alpha=0.03, color=PALETTE["line_primary"], s=5, edgecolors="none")

    n_bins = 40
    bin_edges = np.linspace(final_spots.min(), final_spots.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_mean = np.full(n_bins, np.nan)
    bin_p10 = np.full(n_bins, np.nan)
    bin_p90 = np.full(n_bins, np.nan)

    for j in range(n_bins):
        mask = (final_spots >= bin_edges[j]) & (final_spots < bin_edges[j + 1])
        if mask.sum() > 0:
            bp = pnls[mask]
            bin_mean[j] = np.mean(bp)
            bin_p10[j] = np.percentile(bp, 10)
            bin_p90[j] = np.percentile(bp, 90)

    valid = ~np.isnan(bin_mean)
    bc = bin_centers[valid]
    ax.fill_between(bc, bin_p10[valid], bin_p90[valid],
                    color=PALETTE["fill_ci"], label="10th–90th pctile")
    ax.plot(bc, bin_mean[valid], color=PALETTE["line_primary"], linewidth=2.5, label="Binned mean P&L")

    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axvline(strike, color=PALETTE["line_strike"], linewidth=1.2, linestyle="--",
               alpha=0.7, label=f"Strike K={strike:.0f}")

    ax.set_title("P&L vs Final Spot Price")
    ax.set_xlabel("Final Spot Price")
    ax.set_ylabel("P&L")
    ax.legend()
    _strip_spines(ax)
    _finalize(fig, "pnl_vs_final_spot.png")


def plot_greek_curves(
    S0: float, K: float, T: float, r: float, sigma: float, is_call: bool
) -> None:
    apply_theme()
    S_range = np.linspace(0.5 * S0, 1.5 * S0, 200)
    greek_names = ["delta", "gamma", "vega", "theta", "rho", "vanna"]
    greeks_data = {g: np.zeros(len(S_range)) for g in greek_names}

    for idx, S in enumerate(S_range):
        res = bs_price_and_greeks(float(S), K, T, r, sigma, is_call)
        for g in greek_names:
            greeks_data[g][idx] = res.get(g, 0.0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    atm_low, atm_high = K * 0.95, K * 1.05

    for i, g in enumerate(greek_names):
        ax = axes[i]
        ax.plot(S_range, greeks_data[g], color=PALETTE["line_primary"], linewidth=2)
        ax.axvline(K, color=PALETTE["line_strike"], linestyle="--", linewidth=1.2, alpha=0.8,
                   label=f"K={K:.0f}")
        ax.axvspan(atm_low, atm_high, color=PALETTE["fill_atm"], label="ATM ±5%")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(g.capitalize())
        ax.set_xlabel("Spot Price")
        ax.set_ylabel("Value")
        _strip_spines(ax)
        if i == 0:
            ax.legend(fontsize=8)

    option_type = "Call" if is_call else "Put"
    fig.suptitle(
        f"{option_type} Greeks | S₀={S0:.0f}, K={K:.0f}, T={T:.2f}y, "
        f"r={r:.2%}, σ={sigma:.2%}",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    _finalize(fig, "greek_curves.png")


def plot_greek_bar(greeks: dict) -> None:
    apply_theme()
    fig, ax = plt.subplots(figsize=FIGSIZE["2d"])

    greek_keys = ["delta", "gamma", "vega", "theta", "rho"]
    labels, values = [], []
    bar_colors = [
        PALETTE["line_primary"], PALETTE["line_mean"], PALETTE["bar_positive"],
        PALETTE["bar_negative"], PALETTE["line_strike"],
    ]

    for key in greek_keys:
        if key in greeks:
            labels.append(key.capitalize())
            values.append(greeks[key])

    bars = ax.bar(labels, values, color=bar_colors[: len(labels)], alpha=0.75, edgecolor="black")

    for bar in bars:
        yval = bar.get_height()
        offset = 0.01 * max(abs(min(values)), abs(max(values))) if values else 0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + (offset if yval >= 0 else -offset),
            f"{yval:.4f}",
            ha="center",
            va="bottom" if yval >= 0 else "top",
            fontsize=9,
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Option Greeks")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    _strip_spines(ax)
    _finalize(fig, "greek_bar.png")


def plot_pnl_attribution(attribution_dict: dict) -> None:
    apply_theme()
    fig, ax = plt.subplots(figsize=(10, 7))

    keys_labels = [
        ("theta_pnl", "Theta"),
        ("gamma_pnl", "Gamma"),
        ("vega_pnl", "Vega"),
        ("vanna_pnl", "Vanna"),
        ("volga_pnl", "Volga"),
        ("cost_pnl", "Costs"),
        ("residual_pnl", "Residual"),
    ]

    means, ylabels = [], []
    for key, label in keys_labels:
        val = float(np.mean(attribution_dict[key]))
        if key == "cost_pnl":
            val = -val
        means.append(val)
        ylabels.append(label)

    total_mean = float(np.mean(attribution_dict["total_pnl"]))
    colors = [PALETTE["bar_positive"] if v >= 0 else PALETTE["bar_negative"] for v in means]
    bars = ax.barh(ylabels, means, color=colors, alpha=0.75, edgecolor="black", height=0.6)

    x_scale = max(abs(min(means, default=0.0)), abs(max(means, default=0.0)), abs(total_mean))
    offset = 0.002 * x_scale if x_scale > 0 else 0.001
    for bar, val in zip(bars, means):
        ax.text(
            val + (offset if val >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
        )

    ax.axvline(total_mean, color=PALETTE["line_mean"], linewidth=2, linestyle="--",
               label=f"Total Mean P&L: {total_mean:.4f}")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("P&L Attribution (Mean per Path)")
    ax.set_xlabel("Mean P&L Contribution")
    ax.legend()
    _strip_spines(ax)
    _finalize(fig, "pnl_attribution.png")


def plot_mc_convergence(
    cum_prices: np.ndarray,
    cum_stderr: np.ndarray,
    analytical_price: float | None = None,
) -> None:
    apply_theme()
    fig, ax = plt.subplots(figsize=FIGSIZE["2d"])

    n = np.arange(1, len(cum_prices) + 1)
    ci_upper = cum_prices + 1.96 * cum_stderr
    ci_lower = cum_prices - 1.96 * cum_stderr

    ax.fill_between(n, ci_lower, ci_upper, color=PALETTE["fill_ci"], label="95% CI")
    ax.plot(n, cum_prices, color=PALETTE["line_primary"], linewidth=2, label="MC Estimate")

    if analytical_price is not None:
        ax.axhline(analytical_price, color=PALETTE["line_bs"], linestyle="--", linewidth=1.5,
                   label=f"BS Analytical: {analytical_price:.4f}")

    ax.set_title("Monte Carlo Convergence")
    ax.set_xlabel("Number of Paths")
    ax.set_ylabel("Option Price")
    ax.legend()
    _strip_spines(ax)
    _finalize(fig, "mc_convergence.png")


def plot_hedging_paths(cumulative_pnls_matrix: np.ndarray) -> None:
    apply_theme()
    n_steps, n_paths = cumulative_pnls_matrix.shape
    fig, ax = plt.subplots(figsize=FIGSIZE["2d"])

    x = np.arange(n_steps)
    n_sample = min(25, n_paths)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n_paths, size=n_sample, replace=False)

    for i in sample_idx:
        ax.plot(x, cumulative_pnls_matrix[:, i],
                color=PALETTE["path_individual"], alpha=PALETTE["path_alpha"], linewidth=0.8)

    mean_path = np.mean(cumulative_pnls_matrix, axis=1)
    p5 = np.percentile(cumulative_pnls_matrix, 5, axis=1)
    p95 = np.percentile(cumulative_pnls_matrix, 95, axis=1)

    ax.fill_between(x, p5, p95, color=PALETTE["fill_ci"], label="5th–95th pctile")
    ax.plot(x, mean_path, color=PALETTE["line_mean"], linewidth=2.5, label="Mean P&L path")
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)

    ax.set_title("Hedging P&L Paths")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Cumulative P&L")
    ax.legend()
    _strip_spines(ax)
    _finalize(fig, "hedging_paths.png")


def plot_stress_scenario_comparison(results_dict: dict) -> None:
    apply_theme()

    scenario_labels = {
        "gaussian": "Gaussian",
        "student_t": "Student-t",
        "spot_vol_shock": "Spot/Vol Shock",
        "short_convexity": "Short Convexity",
    }
    ordered = ["gaussian", "student_t", "spot_vol_shock", "short_convexity"]
    scenarios = [s for s in ordered if s in results_dict]
    labels = [scenario_labels.get(s, s) for s in scenarios]
    var_vals = [results_dict[s]["var_95"] for s in scenarios]
    es_vals = [results_dict[s]["es_95"] for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, var_vals, width, label="VaR 95%",
                   color=PALETTE["bar_var"], alpha=0.8, edgecolor="black")
    bars2 = ax.bar(x + width / 2, es_vals, width, label="ES 95%",
                   color=PALETTE["bar_es"], alpha=0.8, edgecolor="black")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                f"{h:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Stress Scenario Comparison — VaR & ES (95%)")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Risk Measure (Loss Magnitude)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    _strip_spines(ax)
    _finalize(fig, "stress_scenario_comparison.png")


def plot_payoff_diagram(S0: float, K: float, premium: float, is_call: bool) -> None:
    apply_theme()
    S_range = np.linspace(K * 0.5, K * 1.5, 200)

    if is_call:
        intrinsic = np.maximum(S_range - K, 0.0)
        breakeven = K + premium
    else:
        intrinsic = np.maximum(K - S_range, 0.0)
        breakeven = K - premium

    net_pnl = intrinsic - premium

    fig, ax = plt.subplots(figsize=FIGSIZE["2d"])
    ax.fill_between(S_range, net_pnl, 0,
                    where=net_pnl > 0, alpha=0.2, color=PALETTE["fill_profit"], label="Profit region")
    ax.fill_between(S_range, net_pnl, 0,
                    where=net_pnl <= 0, alpha=0.2, color=PALETTE["fill_loss"], label="Loss region")

    ax.plot(S_range, intrinsic, color=PALETTE["line_bs"], linewidth=2, linestyle="--",
            label="Intrinsic value")
    ax.plot(S_range, net_pnl, color=PALETTE["line_primary"], linewidth=2.5,
            label="Net P&L (after premium)")
    ax.axhline(0, color="black", linewidth=1, alpha=0.7)
    ax.axvline(K, color=PALETTE["line_strike"], linestyle=":", linewidth=1.2, alpha=0.8,
               label=f"Strike K={K:.0f}")
    ax.scatter([breakeven], [0], color=PALETTE["bar_negative"], s=60, zorder=5)
    ax.annotate(
        f"Breakeven: {breakeven:.2f}",
        xy=(breakeven, 0),
        xytext=(10, 20),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )

    option_type = "Call" if is_call else "Put"
    ax.set_title(f"{option_type} Payoff Diagram | K={K:.0f}, Premium={premium:.4f}")
    ax.set_xlabel("Spot Price at Expiry")
    ax.set_ylabel("Profit / Loss")
    ax.legend()
    _strip_spines(ax)
    _finalize(fig, "payoff_diagram.png")
