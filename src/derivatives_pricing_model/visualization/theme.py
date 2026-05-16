"""Global visualization theme — Deribit/viridis-inspired style with save infrastructure."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

SAVE_DIR: str | None = None
PDF_OUTPUT = None  # type: matplotlib.backends.backend_pdf.PdfPages | None

PALETTE = {
    # colormaps
    "cmap_surface":    "viridis",
    "cmap_heatmap":    "viridis",
    "cmap_diverging":  "RdYlGn",
    # line colors
    "line_primary":    "#2d2d3a",
    "line_mean":       "#1a9e6e",
    "line_var":        "#fde725",
    "line_bs":         "#35b779",
    "line_strike":     "#888888",
    # fill colors
    "fill_tail":       "#d4290a",
    "fill_profit":     "#1a9e6e",
    "fill_loss":       "#d4290a",
    "fill_atm":        "#fde72522",
    "fill_ci":         "#44444422",
    # bar colors
    "bar_positive":    "#35b779",
    "bar_negative":    "#d4290a",
    "bar_var":         "#2d2d3a",
    "bar_es":          "#d4290a",
    # backgrounds / grid
    "bg_pane":         "#f9f9f9",
    "bg_figure":       "#ffffff",
    "grid_color":      "#e0e0e0",
    "grid_style":      "--",
    "grid_alpha":      0.8,
    "tick_color":      "#888888",
    "label_color":     "#555555",
    "title_color":     "#2d2d3a",
    # hedging paths
    "path_individual": "#2d2d3a",
    "path_alpha":      0.12,
}

FIGSIZE = {
    "2d": (10, 6),
    "3d": (12, 8),
    "side": (14, 6),
    "grid": (14, 10),
}

_NON_INTERACTIVE = {"agg", "pdf", "ps", "svg", "template", "pgf", "cairo"}


def apply_theme() -> None:
    plt.rcParams.update(
        {
            "font.family":      "sans-serif",
            "axes.titlesize":   14,
            "axes.labelsize":   11,
            "xtick.labelsize":  10,
            "ytick.labelsize":  10,
            "legend.fontsize":  10,
            "axes.grid":        True,
            "grid.color":       PALETTE["grid_color"],
            "grid.linestyle":   PALETTE["grid_style"],
            "grid.alpha":       PALETTE["grid_alpha"],
            "figure.facecolor": PALETTE["bg_figure"],
            "axes.facecolor":   PALETTE["bg_pane"],
            "xtick.color":      PALETTE["tick_color"],
            "ytick.color":      PALETTE["tick_color"],
            "axes.labelcolor":  PALETTE["label_color"],
            "axes.titlecolor":  PALETTE["title_color"],
        }
    )


def _strip_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _finalize(fig, filename: str) -> None:
    if PDF_OUTPUT is not None:
        PDF_OUTPUT.savefig(fig, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return
    if SAVE_DIR is not None:
        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(SAVE_DIR) / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return
    backend = matplotlib.get_backend().lower()
    if any(token in backend for token in _NON_INTERACTIVE):
        plt.close(fig)
        return
    plt.show()
    plt.close(fig)
