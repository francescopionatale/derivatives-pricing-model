import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from quant_derivatives.engines.stress.scenario import calculate_var_es


NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "template", "pgf", "cairo"}


def _finalize_figure(fig=None):
    fig = fig or plt.gcf()
    backend = matplotlib.get_backend().lower()
    if any(token in backend for token in NON_INTERACTIVE_BACKENDS):
        plt.close(fig)
        return
    plt.show()
    plt.close(fig)


def plot_pnl_distribution(pnls: np.ndarray):
    tail = calculate_var_es(pnls, confidence_level=0.95)
    fig = plt.figure(figsize=(10, 6))
    plt.hist(pnls, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.axvline(np.mean(pnls), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(pnls):.2f}')
    plt.axvline(-tail['var'], color='orange', linestyle='dashed', linewidth=2, label=f'VaR 95% threshold: {-tail["var"]:.2f}')

    ci_lower = np.percentile(pnls, 2.5)
    ci_upper = np.percentile(pnls, 97.5)
    plt.axvline(ci_lower, color='green', linestyle='dotted', linewidth=2, label=f'95% CI Lower: {ci_lower:.2f}')
    plt.axvline(ci_upper, color='green', linestyle='dotted', linewidth=2, label=f'95% CI Upper: {ci_upper:.2f}')

    plt.title('P&L Distribution')
    plt.xlabel('P&L')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    _finalize_figure(fig)


def plot_pnl_surface(final_spots: np.ndarray, pnls: np.ndarray):
    """Plots a 2D density surface (contour) of P&L vs Final Spot Price."""
    fig = plt.figure(figsize=(10, 8))
    try:
        import seaborn as sns
    except ImportError:
        h = plt.hist2d(final_spots, pnls, bins=40, cmap='viridis')
        plt.colorbar(h[3], label='Frequency')
        plt.title('P&L vs Final Spot Density Surface (Histogram)')
        plt.grid(True, alpha=0.3)
    else:
        sns.set_theme(style="whitegrid")
        sns.kdeplot(
            x=final_spots,
            y=pnls,
            cmap="viridis",
            fill=True,
            thresh=0.05,
            levels=20,
        )
        plt.title('P&L vs Final Spot Density Surface')

    plt.xlabel('Final Spot Price')
    plt.ylabel('P&L')
    _finalize_figure(fig)


def plot_pnl_3d_surface(final_spots: np.ndarray, pnls: np.ndarray):
    """Plots a 3D surface of the P&L distribution vs Final Spot Price."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    hist, xedges, yedges = np.histogram2d(final_spots, pnls, bins=30)

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")

    surf = ax.plot_surface(X, Y, hist, cmap='viridis', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Frequency')

    ax.set_title('3D P&L vs Final Spot Surface')
    ax.set_xlabel('Final Spot Price')
    ax.set_ylabel('P&L')
    ax.set_zlabel('Frequency')

    _finalize_figure(fig)


def plot_implied_vol_surface(strikes: np.ndarray, maturities: np.ndarray, implied_vols: np.ndarray):
    """Plots the calibrated implied volatility surface as a 3D trisurf."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    try:
        Triangulation(strikes, maturities)
        surf = ax.plot_trisurf(strikes, maturities, implied_vols, cmap='viridis', edgecolor='none', alpha=0.8)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Implied Volatility')
    except (ValueError, RuntimeError):
        scatter = ax.scatter(strikes, maturities, implied_vols, c=implied_vols, cmap='viridis', s=50)
        fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, label='Implied Volatility')

    ax.set_title('Implied Volatility Surface')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Maturity (Years)')
    ax.set_zlabel('Implied Volatility')

    _finalize_figure(fig)


def plot_implied_vol_smile(strikes: np.ndarray, maturities: np.ndarray, implied_vols: np.ndarray):
    """Plots the implied volatility smile/skew for different maturities."""
    fig = plt.figure(figsize=(10, 6))

    unique_maturities = np.unique(maturities)

    for T in unique_maturities:
        mask = maturities == T
        T_strikes = strikes[mask]
        T_vols = implied_vols[mask]

        sorted_indices = np.argsort(T_strikes)
        T_strikes = T_strikes[sorted_indices]
        T_vols = T_vols[sorted_indices]

        plt.plot(T_strikes, T_vols, marker='o', linestyle='-', label=f'T={T:.2f}')

    plt.title('Implied Volatility Smile / Skew')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.legend(title='Maturity (Years)')
    plt.grid(True, alpha=0.3)
    _finalize_figure(fig)


def plot_pnl_comparison(pnls_base: np.ndarray, pnls_stress: np.ndarray):
    """Plots a comparison of P&L distributions from two different scenarios."""
    fig = plt.figure(figsize=(10, 6))

    plt.hist(pnls_base, bins=50, alpha=0.5, color='blue', edgecolor='black', label='Baseline (Gaussian)', density=True)
    plt.hist(pnls_stress, bins=50, alpha=0.5, color='red', edgecolor='black', label='Stress (Student-t)', density=True)

    plt.axvline(np.mean(pnls_base), color='blue', linestyle='dashed', linewidth=2, label=f'Base Mean: {np.mean(pnls_base):.2f}')
    plt.axvline(np.mean(pnls_stress), color='red', linestyle='dashed', linewidth=2, label=f'Stress Mean: {np.mean(pnls_stress):.2f}')

    plt.title('P&L Distribution Comparison (Baseline vs Stress)')
    plt.xlabel('P&L')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    _finalize_figure(fig)


def plot_pnl_comparison_side_by_side(pnls_base: np.ndarray, pnls_stress: np.ndarray):
    """Plots a side-by-side comparison of P&L distributions from two different scenarios."""
    base_tail = calculate_var_es(pnls_base, confidence_level=0.95)
    stress_tail = calculate_var_es(pnls_stress, confidence_level=0.95)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True, sharex=True)

    ax1.hist(pnls_base, bins=50, alpha=0.75, color='blue', edgecolor='black', density=True)
    ax1.axvline(np.mean(pnls_base), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(pnls_base):.2f}')
    ax1.axvline(-base_tail['var'], color='orange', linestyle='dashed', linewidth=2, label=f'VaR 95% threshold: {-base_tail["var"]:.2f}')
    ax1.set_title('Baseline (Gaussian) P&L')
    ax1.set_xlabel('P&L')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(pnls_stress, bins=50, alpha=0.75, color='red', edgecolor='black', density=True)
    ax2.axvline(np.mean(pnls_stress), color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(pnls_stress):.2f}')
    ax2.axvline(-stress_tail['var'], color='orange', linestyle='dashed', linewidth=2, label=f'VaR 95% threshold: {-stress_tail["var"]:.2f}')
    ax2.set_title('Stress (Student-t) P&L')
    ax2.set_xlabel('P&L')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Side-by-Side P&L Distribution Comparison', fontsize=16)
    plt.tight_layout()
    _finalize_figure(fig)


def plot_historical_volatility(dates, vols, window: int):
    """Plots historical rolling volatility."""
    fig = plt.figure(figsize=(12, 6))
    plt.plot(dates, vols, color='purple', linewidth=2, label=f'{window}-Day Rolling Volatility')

    plt.title('Historical Realized Volatility')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _finalize_figure(fig)


def plot_pnl_vs_initial_spot(s0_values: np.ndarray, pnls: np.ndarray):
    """Plots a scatter plot of P&L against the initial spot price (S0)."""
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(s0_values, pnls, alpha=0.5, color='teal', edgecolor='black', s=10)
    plt.title('P&L vs Initial Spot Price (S0)')
    plt.xlabel('Initial Spot Price (S0)')
    plt.ylabel('P&L')
    plt.grid(True, alpha=0.3)
    _finalize_figure(fig)


def plot_greeks(greeks: dict):
    """Plots the calculated Greeks as a bar chart."""
    fig = plt.figure(figsize=(10, 6))

    greek_keys = ['delta', 'gamma', 'vega', 'theta', 'rho']
    labels = []
    values = []

    for key in greek_keys:
        if key in greeks:
            labels.append(key.capitalize())
            values.append(greeks[key])

    bars = plt.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], alpha=0.7, edgecolor='black')

    for bar in bars:
        yval = bar.get_height()
        offset = 0.01 * max(abs(min(values)), abs(max(values))) if values else 0.01
        plt.text(bar.get_x() + bar.get_width() / 2, yval + (offset if yval >= 0 else -offset),
                 f'{yval:.4f}', ha='center', va='bottom' if yval >= 0 else 'top')

    plt.title('Option Greeks')
    plt.ylabel('Value')
    plt.axhline(0, color='black', linewidth=1)
    plt.grid(axis='y', alpha=0.3)
    _finalize_figure(fig)
