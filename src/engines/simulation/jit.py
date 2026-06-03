"""Optional Numba JIT acceleration for hot simulation loops.

Numba is listed under the [accel] optional extra and has no wheel for CPython 3.14.
The module provides a `maybe_jit` decorator that falls back to a no-op when Numba
is not installed, so all callsites work identically with or without acceleration.

Usage::

    from engines.simulation.jit import maybe_jit

    @maybe_jit
    def _inner_loop(...):
        ...
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])

try:
    from numba import njit as _numba_njit  # type: ignore[import-not-found]

    NUMBA_AVAILABLE = True

    def maybe_jit(fn: F) -> F:
        """Wrap fn with numba.njit (nopython mode, cache=True)."""
        return _numba_njit(cache=True)(fn)  # type: ignore[return-value]

except ImportError:
    NUMBA_AVAILABLE = False

    def maybe_jit(fn: F) -> F:  # type: ignore[misc]
        """No-op decorator: returns fn unchanged when Numba is unavailable."""
        return fn


# ---------------------------------------------------------------------------
# JIT-accelerated inner loops (pure numpy fallbacks when numba not available)
# ---------------------------------------------------------------------------

@maybe_jit
def _gbm_step_loop(
    paths: np.ndarray,
    Z: np.ndarray,
    drift: float,
    vol_sqrt_dt: float,
    n_steps: int,
) -> None:
    """In-place GBM stepping: paths[t+1] = paths[t] * exp(drift + vol*sqrt(dt)*Z[t])."""
    for t in range(n_steps):
        paths[t + 1] = paths[t] * np.exp(drift + vol_sqrt_dt * Z[t])


@maybe_jit
def _heston_euler_step_loop(
    S: np.ndarray,
    V: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    kappa: float,
    theta: float,
    sigma_v: float,
    r: float,
    dt: float,
    sqrt_dt: float,
    n_steps: int,
) -> None:
    """In-place Heston Euler-Maruyama time-stepping."""
    for t in range(1, n_steps + 1):
        v_prev = np.maximum(V[t - 1], 0.0)
        V[t] = v_prev + kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev) * sqrt_dt * W2[t - 1]
        S[t] = S[t - 1] * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev) * sqrt_dt * W1[t - 1])


def gbm_step_loop(
    paths: np.ndarray,
    Z: np.ndarray,
    drift: float,
    vol_sqrt_dt: float,
    n_steps: int,
) -> None:
    """Public GBM stepping, dispatches to JIT version when available."""
    _gbm_step_loop(paths, Z, drift, vol_sqrt_dt, n_steps)


def heston_euler_step_loop(
    S: np.ndarray,
    V: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    kappa: float,
    theta: float,
    sigma_v: float,
    r: float,
    dt: float,
    sqrt_dt: float,
    n_steps: int,
) -> None:
    """Public Heston Euler stepping, dispatches to JIT version when available."""
    _heston_euler_step_loop(S, V, W1, W2, kappa, theta, sigma_v, r, dt, sqrt_dt, n_steps)
