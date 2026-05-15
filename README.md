# Derivatives Pricing Model

A quantitative finance toolkit for derivatives pricing, stochastic-volatility simulation, hedging, stress testing, volatility calibration, exotic options, and portfolio risk optimization.

The `pyproject.toml` file is the source of truth for packaging and installation metadata; `requirements.txt` mirrors the same runtime dependencies for convenience.

## Features

- **Pricing Engines**: Black-Scholes, Binomial Tree (CRR), Monte Carlo, Heston Monte Carlo, barrier options, and lookback options.
- **Hedging Simulation**: Discrete-time delta hedging with P&L attribution across theta, gamma, vega, vanna, volga, transaction costs, and residual slippage.
- **Stress Testing**: Gaussian, heavy-tail, finite spot/vol shock, and short-convexity scenarios with VaR and Expected Shortfall.
- **Calibration**: Implied volatility calculation, surface smoothing, static no-arbitrage checks, and Heston parameter calibration to vanilla quotes.
- **Risk Optimization**: Portfolio hedge optimization under linear neutrality constraints and quadratic residual-risk penalties.
- **Terminal-first Output**: Results are printed in the terminal; plots are shown only when the backend is interactive.

## CLI Reference

The full CLI reference documenting all commands, arguments, and architecture is available in [`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md).

Quick overview of available commands:

| Command | Description | Engine |
| --- | --- | --- |
| `bs-price` | Black-Scholes analytical pricing + Greeks | `bs_price_and_greeks` |
| `binomial-price` | Binomial tree (CRR) pricing | `binomial_price` |
| `mc-price` | Monte Carlo pricing under GBM | `PricingWorkflow.run_mc` |
| `heston-price` | Monte Carlo pricing under Heston SV | `heston_vanilla_price_mc` |
| `barrier-price` | Barrier option pricing (GBM or Heston) | `price_barrier_mc` / `price_barrier_heston_mc` |
| `lookback-price` | Lookback option pricing (GBM or Heston) | `price_lookback_mc` / `price_lookback_heston_mc` |
| `calibrate-surface` | IV surface + no-arbitrage checks | `calibrate_surface_with_smoothing` |
| `calibrate-heston` | Heston parameter fitting to vanilla quotes | `calibrate_heston_to_quotes` |
| `hedge-sim` | Discrete delta hedging simulation | `simulate_discrete_hedging` |
| `stress-run` | Stress testing with VaR/ES | `calculate_var_es` |
| `optimize-risk` | Portfolio Greek neutrality optimization | `optimize_portfolio` |
| `hist-vol` | Historical realized volatility | `HistoricalVolWorkflow.run` |

## Installation

Recommended installation for normal use and for the console command:

```bash
pip install .
```

For development mode:

```bash
pip install -e .
```

If you only want the raw Python dependencies without installing the package metadata, you can still use the matching runtime dependency set:

```bash
pip install -r requirements.txt
```

After package installation, the console entry point is available as:

```bash
derivatives-pricing-model --help
```

## Usage

### Pricing
```bash
# Black-Scholes call (default)
python main.py bs-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2

# The installed console script works the same way
derivatives-pricing-model bs-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2

# Black-Scholes put
python main.py bs-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --put

# Monte Carlo call (default)
python main.py mc-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --M 10000 --n-steps 252 --seed 42 --antithetic

# Heston Monte Carlo pricing with explicit parameters
python main.py heston-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --M 10000 --n-steps 252 --kappa 2.0 --theta 0.04 --sigma-v 0.30 --rho -0.70 --v0 0.04 --antithetic

# Heston Monte Carlo pricing using calibrated parameters from JSON
python main.py heston-price --S0 100 --K 100 --T 1.0 --r 0.03 --M 10000 --n-steps 252 --params-json examples/heston/reference_params.json --antithetic

# Up-and-out barrier call under GBM
python main.py barrier-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --barrier 120 --direction up --barrier-style out --M 10000 --n-steps 252 --seed 42 --antithetic

# Up-and-out barrier call under calibrated Heston
python main.py barrier-price --model heston --params-json examples/heston/reference_params.json --S0 100 --K 100 --T 1.0 --r 0.03 --barrier 120 --direction up --barrier-style out --M 10000 --n-steps 252 --seed 42

# Floating-strike lookback call under calibrated Heston
python main.py lookback-price --model heston --params-json examples/heston/reference_params.json --S0 100 --K 100 --T 1.0 --r 0.03 --lookback-style floating --M 10000 --n-steps 252 --seed 42
```

### Hedging Simulation
```bash
# GBM baseline vs Student-t stress
python main.py hedge-sim --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --n-steps 252 --M 1000 --cost 0.001 --seed 42

# Heston baseline vs stressed stochastic-volatility scenario
python main.py hedge-sim --model heston --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --n-steps 252 --M 1000 --cost 0.001 --kappa 2.0 --theta 0.04 --sigma-v 0.30 --rho -0.70 --v0 0.04 --seed 42
```

### Stress Testing
```bash
python main.py stress-run --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --n-steps 252 --M 1000 --df 4.0 --spot-shock -0.10 --vol-shock 0.05 --seed 42
```

### Calibration
```bash
# Static no-arbitrage and implied-volatility surface
python main.py calibrate-surface --input-csv quotes.csv --S0 100 --r 0.05

# Heston calibration from market quotes, with optional JSON export
python main.py calibrate-heston --input-csv examples/heston/synthetic_heston_quotes.csv --S0 100 --r 0.03 --M 4000 --n-steps 64 --maxiter 30 --antithetic --output-json examples/heston/calibrated_params.json
```

### Portfolio Risk Optimization
```bash
python main.py optimize-risk --input-json examples/risk/portfolio_case.json
```

## Heston calibration and exotic pipeline

A practical workflow is available directly from the CLI:

1. **Calibrate Heston parameters** from vanilla option quotes.
2. **Save the calibrated parameter set** to JSON.
3. **Reuse the same parameter set** for vanilla Heston pricing, barrier pricing, and lookback pricing.

Relevant example files:

- `examples/heston/synthetic_heston_quotes.csv`
- `examples/heston/reference_params.json`

The `--params-json` flag is supported by:

- `heston-price`
- `barrier-price --model heston`
- `lookback-price --model heston`

Notes:

- `heston-price` is always a Heston command and therefore does **not** accept `--model`.
- When using Heston-based commands with `--params-json`, `--sigma` is optional and omitted from the examples above. For GBM commands and GBM mode it remains required.
- `make test` runs both unit and integration tests. `make lint` performs Python-only repository checks and no longer depends on a removed frontend toolchain.

## Project Structure

- `src/derivatives_pricing_model/engines/`: Core mathematical models and calibration engines.
- `src/derivatives_pricing_model/workflows/`: Orchestration of pricing, hedging, stress, calibration, and optimization tasks.
- `src/derivatives_pricing_model/utils/`: Logging, validation, and Heston-parameter helpers.
- `examples/`: Runnable command examples and sample calibration/optimization payloads.
- `tests/`: Unit and integration tests.
