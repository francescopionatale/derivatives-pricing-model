# CLI Reference

The CLI is the primary entry point for the derivatives-pricing-model toolkit. It uses Python's `argparse` with a sub-command dispatch pattern.

## Architecture

```
derivatives-pricing-model <command> [options]
```

Each CLI command maps to a **Workflow** (orchestration layer) that calls one or more **Engines** (core math):

```
CLI command
  └→ Workflow (coordinates data loading, engine calls, output formatting)
       └→ Engine (pure mathematical computation, no I/O)
```

### Argument Parsing Infrastructure

The CLI uses shared utility functions in `src/derivatives_pricing_model/cli/parser.py` to inject standard argument groups:

| Utility Function | Purpose | Key Arguments |
| --- | --- | --- |
| `add_standard_market_args` | Standard Black-Scholes inputs | `--S0`, `--K`, `--T`, `--r`, `--sigma`, `--put` |
| `add_heston_args` | Parameters for Heston MC simulation | `--kappa`, `--theta`, `--sigma-v`, `--rho`, `--v0`, `--params-json`, `--antithetic` |
| `add_heston_calibration_args` | Settings for the Heston optimizer | `--M`, `--n-steps`, `--maxiter`, `--weight-mode`, `--output-json` |

---

## Pricing Commands

### `bs-price` — Black-Scholes Pricing

Calculates the analytical closed-form price and Greeks (Delta, Gamma, Vega, Theta, Rho, Vanna, Volga) for European options.

**Arguments:** Standard market args + `--target-price` (optional, triggers implied volatility calculation).

**Workflow:** `PricingWorkflow.run_bs()`

```bash
derivatives-pricing-model bs-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2
derivatives-pricing-model bs-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --put
```

---

### `binomial-price` — Binomial Tree (CRR) Pricing

Prices European options using the Cox-Ross-Rubinstein binomial model.

**Arguments:** Standard market args + `--n-steps` (default: 100).

**Workflow:** `PricingWorkflow.run_binomial()`

```bash
derivatives-pricing-model binomial-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --n-steps 1000
```

---

### `mc-price` — Monte Carlo Pricing (GBM)

Prices European options using Geometric Brownian Motion simulation.

**Arguments:** Standard market args + `--M` (paths, default: 10000), `--n-steps` (default: 252), `--seed`, `--antithetic`.

**Workflow:** `PricingWorkflow.run_mc()`

```bash
derivatives-pricing-model mc-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --M 10000 --n-steps 252 --seed 42 --antithetic
```

---

### `heston-price` — Heston Stochastic Volatility Pricing

Prices European options under the Heston model via Monte Carlo. This command is always Heston-based and does not accept `--model`.

**Arguments:** Standard market args (sigma optional) + Heston args + `--M`, `--n-steps`, `--seed`.

**Workflow:** `PricingWorkflow.run_heston()`

```bash
# With explicit parameters
derivatives-pricing-model heston-price --S0 100 --K 100 --T 1.0 --r 0.05 --M 10000 --n-steps 252 --kappa 2.0 --theta 0.04 --sigma-v 0.30 --rho -0.70 --v0 0.04 --antithetic

# With calibrated parameters from JSON
derivatives-pricing-model heston-price --S0 100 --K 100 --T 1.0 --r 0.03 --M 10000 --n-steps 252 --params-json examples/heston/reference_params.json --antithetic
```

---

### `barrier-price` — Barrier Option Pricing

Prices path-dependent barrier options (Up/Down, In/Out). GBM uses Brownian bridge correction; Heston uses discrete path monitoring.

**Arguments:** Standard market args + `--barrier`, `--direction` (up/down), `--barrier-style` (in/out) + `--model` (gbm/heston) + Heston args if applicable.

**Workflow:** `PricingWorkflow.run_barrier()`

```bash
# GBM with Brownian bridge correction
derivatives-pricing-model barrier-price --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --barrier 120 --direction up --barrier-style out --M 10000 --n-steps 252 --seed 42 --antithetic

# Heston with calibrated parameters
derivatives-pricing-model barrier-price --model heston --params-json examples/heston/reference_params.json --S0 100 --K 100 --T 1.0 --r 0.03 --barrier 120 --direction up --barrier-style out --M 10000 --n-steps 252 --seed 42
```

---

### `lookback-price` — Lookback Option Pricing

Prices lookback options where the payoff depends on path extrema. Supports fixed-strike and floating-strike under GBM (Brownian bridge extrema correction) and Heston (discrete monitoring).

**Arguments:** Standard market args + `--lookback-style` (fixed/floating) + `--model` (gbm/heston) + Heston args if applicable.

**Workflow:** `PricingWorkflow.run_lookback()`

```bash
# Floating-strike lookback call under Heston
derivatives-pricing-model lookback-price --model heston --params-json examples/heston/reference_params.json --S0 100 --K 100 --T 1.0 --r 0.03 --lookback-style floating --M 10000 --n-steps 252 --seed 42
```

---

## Calibration Commands

### `calibrate-surface` — Implied Volatility Surface

Calibrates an implied volatility surface from market quotes. Performs static no-arbitrage checks (butterfly and calendar spread violations) and produces a smoothed volatility grid.

**Arguments:** `--input-csv`, `--S0`, `--r`.

**Workflow:** `CalibrationWorkflow.run()`

```bash
derivatives-pricing-model calibrate-surface --input-csv quotes.csv --S0 100 --r 0.05
```

---

### `calibrate-heston` — Heston Parameter Fitting

Calibrates the five Heston parameters (κ, θ, σᵥ, ρ, v₀) to vanilla option quotes using L-BFGS-B optimization with a soft Feller condition penalty. Outputs RMSE on prices and implied volatilities.

**Arguments:** `--input-csv`, `--S0`, `--r` + calibration args + initial guesses (`--init-kappa`, `--init-theta`, `--init-sigma-v`, `--init-rho`, `--init-v0`).

**Workflow:** `CalibrationWorkflow.run_heston_calibration()`

```bash
derivatives-pricing-model calibrate-heston --input-csv examples/heston/synthetic_heston_quotes.csv --S0 100 --r 0.03 --M 4000 --n-steps 64 --maxiter 30 --antithetic --output-json examples/heston/calibrated_params.json
```

---

## Hedging & Risk Commands

### `hedge-sim` — Discrete Delta Hedging Simulation

Simulates discrete-time delta hedging over multiple paths with P&L attribution across theta, gamma, vega, vanna, volga, transaction costs, and residual slippage.

**Arguments:** Standard market args + `--M`, `--n-steps`, `--cost` (proportional transaction cost) + `--model` (gbm/heston) + Heston args if applicable.

**Workflow:** `HedgingWorkflow.run()`

```bash
# GBM
derivatives-pricing-model hedge-sim --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --n-steps 252 --M 1000 --cost 0.001 --seed 42

# Heston
derivatives-pricing-model hedge-sim --model heston --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --n-steps 252 --M 1000 --cost 0.001 --kappa 2.0 --theta 0.04 --sigma-v 0.30 --rho -0.70 --v0 0.04 --seed 42
```

---

### `stress-run` — Stress Testing

Evaluates portfolio performance under stress scenarios: Gaussian baseline, heavy-tailed (Student-t), and finite spot/vol shocks. Reports VaR and Expected Shortfall.

**Arguments:** Standard market args + `--M`, `--n-steps`, `--df` (degrees of freedom for Student-t), `--spot-shock`, `--vol-shock`, `--seed`.

**Workflow:** `StressWorkflow.run()`

```bash
derivatives-pricing-model stress-run --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --n-steps 252 --M 1000 --df 4.0 --spot-shock -0.10 --vol-shock 0.05 --seed 42
```

---

### `optimize-risk` — Portfolio Risk Optimization

Optimizes portfolio hedging under linear Greek neutrality constraints and quadratic residual-risk penalties using SLSQP.

**Arguments:** `--input-json` (portfolio config with `current_greeks`, `available_instruments`, `factor_covariance`).

**Workflow:** `RiskWorkflow.run()`

```bash
derivatives-pricing-model optimize-risk --input-json examples/risk/portfolio_case.json
```

---

### `hist-vol` — Historical Volatility

Calculates annualized rolling realized volatility from historical price data.

**Arguments:** `--input-csv`, `--window` (rolling window in days, default: 21), `--date-col`, `--price-col`.

**Workflow:** `HistoricalVolWorkflow.run()`

```bash
derivatives-pricing-model hist-vol --input-csv prices.csv --window 21
```

---

## Heston Calibration → Exotic Pricing Pipeline

A typical end-to-end workflow:

**Step 1 — Calibrate** Heston parameters from vanilla quotes:
```bash
derivatives-pricing-model calibrate-heston --input-csv quotes.csv --S0 100 --r 0.03 --M 4000 --n-steps 64 --maxiter 30 --antithetic --output-json params.json
```

**Step 2 — Price vanillas** with calibrated parameters:
```bash
derivatives-pricing-model heston-price --S0 100 --K 100 --T 1.0 --r 0.03 --params-json params.json --M 50000 --n-steps 252 --antithetic
```

**Step 3 — Price exotics** with the same parameters:
```bash
derivatives-pricing-model barrier-price --model heston --params-json params.json --S0 100 --K 100 --T 1.0 --r 0.03 --barrier 120 --direction up --barrier-style out --M 50000 --n-steps 252

derivatives-pricing-model lookback-price --model heston --params-json params.json --S0 100 --K 100 --T 1.0 --r 0.03 --lookback-style floating --M 50000 --n-steps 252
```

The `--params-json` flag is supported by `heston-price`, `barrier-price --model heston`, and `lookback-price --model heston`.
