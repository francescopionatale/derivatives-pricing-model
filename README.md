# Derivatives Pricing Model

A quantitative finance toolkit for derivatives pricing, stochastic-volatility simulation, hedging, stress testing, volatility calibration, exotic options, and portfolio risk optimization — built to quant-recruiting standards (Citadel / Jane Street / Two Sigma / DE Shaw level).

The `pyproject.toml` file is the source of truth for packaging and installation metadata; `requirements.txt` mirrors the same runtime dependencies for convenience.

## Features

### Pricing Engines
- **Black-Scholes** analytical pricing + full Greeks (Δ, Γ, ν, Θ, ρ, vanna, volga)
- **Binomial Tree (CRR)** — European and American options
- **Heston Monte Carlo** — Euler-Maruyama and Andersen QE schemes
- **Heston Fourier** (v0.2) — Carr-Madan FFT, Fang-Oosterlee COS, Lewis quadrature; three methods cross-validate to < 2 cents
- **SABR** (v0.2) — Hagan (2002) lognormal & normal (Bachelier) implied vol; α/ρ/ν calibration
- **Merton Jump-Diffusion** (v0.2) — Poisson-weighted Black-Scholes series; model-agnostic Fourier char function
- **BS PDE Crank-Nicolson** (v0.2) — log-spot θ-scheme, American via payoff clamping, O(Δx²) convergence
- **Heston 2-D ADI PDE** (v0.2) — Douglas-Rachford splitting on (log-S, v) grid; matches Fourier to 1.5% ATM
- **Longstaff-Schwartz American MC** (v0.2) — backward induction, ITM-only regression, exercise boundary, 95% CI
- **Local Volatility (Dupire)** (v0.2) — finite-difference formula in total-variance form; Gatheral SVI calibration and butterfly arbitrage check
- **Barrier & Lookback options** — GBM and Heston path-dependent pricing

### Simulation Infrastructure
- **Variance Reduction** (v0.2) — control variates (β* estimator), Sobol QMC, moment matching; accessible via `simulate_gbm_paths(quasi_mc=True, moment_matching=True)`
- **Reproducible RNG** — `numpy.random.default_rng` + `SeedSequence.spawn` throughout; no global state
- **Optional Numba JIT** (v0.2) — `engines/simulation/jit.py` `maybe_jit` decorator; falls back to pure NumPy

### Calibration
- **Heston calibration** — L-BFGS-B with Feller-condition penalty; default `pricing_method="cos"` (~1000× faster than MC)
- **SABR calibration** — α from ATM cubic, 2-D L-BFGS-B over (ρ, ν)
- **SVI calibration** — Gatheral raw SVI 5-parameter fit; butterfly arbitrage check
- **IV surface smoothing** — static no-arbitrage checks, spline smoothing

### Hedging & Risk
- **Discrete delta hedging** — P&L attribution: θ / Γ / ν / vanna / volga / transaction costs / residual
- **Stress testing** — Gaussian, heavy-tail, finite spot-vol shock, short-convexity; VaR and Expected Shortfall
- **Portfolio optimization** — SLSQP under linear Greek-neutrality constraints

### Testing (218 tests)
- **Unit tests** — per-module, covering all new engines
- **Cross-validation tests** — 3-way Fourier agreement, BS limits, American bounds, put-call parity across all engines
- **Invariant tests** — monotonicity, price non-negativity, delta bounds, SABR smoothness
- **Regression tests** — 16 golden values pinned to guard against silent regressions
- **Benchmark tests** — `pytest-benchmark` suite for BS, COS, MC, PDE

## CLI Reference

The full CLI reference is in [`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md).

Quick command overview:

| Command | Description | Engine |
|---|---|---|
| `bs-price` | Black-Scholes analytical pricing + Greeks | `bs_price_and_greeks` |
| `binomial-price` | Binomial tree (CRR) pricing | `binomial_price` |
| `mc-price` | Monte Carlo under GBM | `simulate_gbm_paths` |
| `heston-price` | Monte Carlo under Heston SV | `heston_vanilla_price_mc` |
| `barrier-price` | Barrier option (GBM or Heston) | `price_barrier_mc` |
| `lookback-price` | Lookback option (GBM or Heston) | `price_lookback_mc` |
| `calibrate-surface` | IV surface + no-arbitrage checks | `calibrate_surface_with_smoothing` |
| `calibrate-heston` | Heston fitting (COS default) | `calibrate_heston_to_quotes` |
| `hedge-sim` | Discrete delta hedging simulation | `simulate_discrete_hedging` |
| `stress-run` | Stress testing with VaR/ES | `calculate_var_es` |
| `optimize-risk` | Portfolio Greek optimization | `optimize_portfolio` |
| `hist-vol` | Historical realized volatility | — |
| `fourier-price` ★ | Heston Fourier (COS/CM/Lewis) | `heston_price_cos` |
| `american-price` ★ | Longstaff-Schwartz American MC | `american_option_lsm` |
| `merton-price` ★ | Merton jump-diffusion series | `merton_price` |
| `heston-pde-price` ★ | Heston 2-D ADI PDE | `heston_pde_price` |
| `sabr-vol` ★ | SABR implied vol (Hagan 2002) | `sabr_implied_vol` |
| `sabr-calibrate` ★ | Calibrate SABR to a smile | `calibrate_sabr` |
| `dupire-surface` ★ | Dupire local vol at (K, T) | `dupire_local_vol` |

*★ = added in v0.2*

## Installation

```bash
pip install -e ".[dev]"      # development (includes pytest, ruff, mypy)
pip install -e ".[accel]"    # optional Numba JIT acceleration (requires compatible CPython)
```

After installation, the console entry point is available as:

```bash
derivatives-pricing-model --help
```

## Quick Start

```bash
# Black-Scholes call
python main.py bs-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2

# Heston Fourier (three methods, cross-validate)
python main.py fourier-price --S0 100 --K 100 --T 1 --r 0.05 \
  --kappa 2.0 --theta 0.04 --sigma-v 0.3 --rho -0.7 --v0 0.04 --method all

# American put via Longstaff-Schwartz
python main.py american-price --S0 100 --K 100 --T 1 --r 0.05 --sigma 0.2 \
  --put --n-paths 50000 --seed 42

# Merton jump-diffusion
python main.py merton-price --S0 100 --K 100 --T 1 --r 0.05 --sigma 0.2 \
  --lam 1.0 --mu-J -0.10 --sigma-J 0.15

# Fast Heston calibration (COS, ~seconds vs minutes with MC)
python main.py calibrate-heston \
  --input-csv examples/heston/synthetic_heston_quotes.csv \
  --S0 100 --r 0.05 --maxiter 300 --output-json params.json
```

## Usage Examples

### Pricing
```bash
python main.py bs-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2
python main.py bs-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --put
python main.py mc-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --M 10000 --n-steps 252 --seed 42 --antithetic
python main.py heston-price --S0 100 --K 100 --T 1.0 --r 0.05 --M 10000 --n-steps 252 \
  --kappa 2.0 --theta 0.04 --sigma-v 0.30 --rho -0.70 --v0 0.04 --antithetic
python main.py barrier-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 \
  --barrier 120 --direction up --barrier-style out --M 10000 --n-steps 252 --seed 42
```

### Hedging & Stress
```bash
python main.py hedge-sim --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 \
  --n-steps 252 --M 1000 --cost 0.001 --seed 42
python main.py stress-run --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 \
  --n-steps 252 --M 1000 --df 4.0 --spot-shock -0.10 --vol-shock 0.05 --seed 42
```

### Calibration
```bash
python main.py calibrate-surface --input-csv quotes.csv --S0 100 --r 0.05
python main.py calibrate-heston \
  --input-csv examples/heston/synthetic_heston_quotes.csv \
  --S0 100 --r 0.03 --maxiter 300 --output-json params.json

# SABR smile calibration
python main.py sabr-calibrate --F 100 --T 1 \
  --strikes "90,95,100,105,110" --vols "0.22,0.21,0.20,0.21,0.22"
```

### Convergence Analysis
```bash
# Generate 8 publication-quality convergence figures in docs/figures/
MPLBACKEND=Agg PYTHONPATH=src python notebooks/convergence_analysis.py
```

## Project Structure

```
src/
├── cli/              # Argument parser, dispatcher, and per-command modules (19 commands)
├── engines/
│   ├── pricing/      # BS, binomial, MC, Heston MC+Fourier, SABR, PDE, American LSM,
│   │                 # Merton jumps, local vol/SVI, barrier/lookback, implied vol
│   ├── simulation/   # GBM, Heston (Euler+QE), Merton, variance reduction, JIT
│   ├── calibration/  # IV surface smoothing, Heston calibration (COS default)
│   ├── hedging/      # Discrete delta hedging
│   ├── stress/       # Scenario generation, VaR/ES
│   └── risk/         # Portfolio Greek optimization
├── workflows/        # Orchestration wrappers
├── visualization/    # Deribit/viridis theme and plot functions
├── utils/            # Logging, validation, helpers
├── data_io/          # CSV/JSON loaders
├── models/           # Domain dataclasses
└── configs/          # Configuration dataclasses
notebooks/            # convergence_analysis.py → docs/figures/
docs/
├── CLI_REFERENCE.md  # Full command reference
└── figures/          # 8 convergence analysis PNGs
examples/             # Sample CSV/JSON for calibration and risk commands
tests/
├── unit/             # Per-module unit tests
├── integration/      # End-to-end CLI tests
├── regression/       # Golden-value regression tests
└── benchmark/        # pytest-benchmark performance suite
```
