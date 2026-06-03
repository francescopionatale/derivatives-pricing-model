"""Shared pytest configuration.

Force a non-interactive matplotlib backend so the visualization layer never tries
to open a display during tests (headless CI, plot-saving code paths).
"""
import matplotlib

matplotlib.use("Agg")
