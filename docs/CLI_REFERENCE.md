# CLI Reference

The CLI is the primary entry point for the `derivatives-pricing-model` toolkit — a quantitative finance library for derivatives pricing, stochastic volatility simulation, hedging, stress testing, and portfolio risk optimization.

---

## Three-Layer Architecture

```mermaid
graph TD
    subgraph CLI["CLI Layer"]
        CMD["CLI Commands\ncli/commands/*.py"]
        PARSER["Argument Parser\ncli/parser.py"]
        MAIN["Dispatcher\ncli/main.py"]
    end

    subgraph WF["Workflows Layer"]
        PW["PricingWorkflow"]
        CW["CalibrationWorkflow"]
        HW["HedgingWorkflow"]
        SW["StressWorkflow"]
        RW["RiskWorkflow"]
        HVW["HistoricalVolWorkflow"]
    end

    subgraph ENG["Engines Layer"]
        SIM["Simulation\ngbm · heston"]
        PRC["Pricing\nbs · binomial · heston · exotics · implied_vol"]
        CAL["Calibration\nsurface · heston"]
        HDG["Hedging\ndiscrete_hedging"]
        STR["Stress\nscenario"]
        RSK["Risk\noptimization"]
    end

    subgraph SUP["Support"]
        IO["data_io\nloaders"]
        VIZ["Visualization\nplots"]
        UTL["Utils\nvalidation · heston_params · logging"]
        MDL["Models\ndomain"]
    end

    CMD --> PARSER
    CMD --> MAIN
    MAIN --> PW & CW & HW & SW & RW & HVW
    PW & CW & HW & SW --> SIM & PRC
    CW --> CAL
    HW & SW --> HDG & STR
    RW --> RSK
    CW & HVW --> IO
    PW & CW & HW & HVW --> VIZ
    SIM & PRC & CAL & HDG & STR & RSK --> UTL
    CAL & IO --> MDL
    VIZ --> STR
```

---

## Command → Workflow → Engine Map

```mermaid
graph LR
    subgraph commands["CLI Commands"]
        bs["bs-price"]
        mc["mc-price"]
        hpx["heston-price"]
        bin["binomial-price"]
        bar["barrier-price"]
        lbk["lookback-price"]
        cs["calibrate-surface"]
        ch["calibrate-heston"]
        hs["hedge-sim"]
        sr["stress-run"]
        optR["optimize-risk"]
        hv["hist-vol"]
    end

    subgraph workflows["Workflows"]
        PW["PricingWorkflow"]
        CW["CalibrationWorkflow"]
        HW["HedgingWorkflow"]
        SW["StressWorkflow"]
        RW["RiskWorkflow"]
        HVW["HistoricalVolWorkflow"]
    end

    subgraph engines["Engines"]
        bse["bs_price_and_greeks"]
        iv["implied_volatility"]
        gbm["simulate_gbm_paths"]
        hmc["heston_vanilla_price_mc"]
        shp["simulate_heston_paths"]
        bip["binomial_price_and_greeks"]
        exo["price_barrier_mc\nprice_barrier_heston_mc\nprice_lookback_mc\nprice_lookback_heston_mc"]
        lqc["load_quotes_csv"]
        ns["check_no_arbitrage\ncalibrate_surface_with_smoothing"]
        chq["calibrate_heston_to_quotes"]
        sdh["simulate_discrete_hedging"]
        ves["calculate_var_es"]
        stu["generate_student_t_paths\napply_spot_vol_shock"]
        op["optimize_portfolio"]
        lhp["load_historical_prices"]
    end

    bs --> PW --> bse & iv
    mc --> PW --> gbm
    hpx --> PW --> hmc --> shp
    bin --> PW --> bip
    bar --> PW --> exo
    lbk --> PW --> exo
    cs --> CW --> lqc & ns
    ch --> CW --> chq --> hmc
    hs --> HW --> gbm & shp & sdh & ves
    sr --> SW --> gbm & shp & stu & sdh & ves
    optR --> RW --> op
    hv --> HVW --> lhp
```

---

## Engine Internal Dependencies

```mermaid
graph TD
    iv["implied_vol.py\nimplied_volatility()"] --> bse["black_scholes.py\nbs_price_and_greeks()"]
    hv_eng["heston_vanilla.py\nheston_vanilla_price_mc()"] --> shp["simulation/heston.py\nsimulate_heston_paths()"]
    exo["exotics.py\nprice_barrier_mc\nprice_lookback_mc"] --> gbm_eng["simulation/gbm.py\nsimulate_gbm_paths()"]
    exo --> shp
    cal_h["calibration/heston.py\ncalibrate_heston_to_quotes()"] --> hv_eng
    cal_h --> iv
    dh["hedging/discrete_hedging.py\nsimulate_discrete_hedging()"] --> bse
    plots["visualization/plots.py\nplot_pnl_distribution()"] --> varES["stress/scenario.py\ncalculate_var_es()"]
```

---

## Data Flow: Market Quotes → Calibration → Exotic Pricing

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant CW as CalibrationWorkflow
    participant Loader as io.loaders
    participant CalHeston as calibration.heston
    participant HestonVanilla as pricing.heston_vanilla
    participant SimHeston as simulation.heston
    participant Exotics as pricing.exotics

    User->>CLI: calibrate-heston --input-csv quotes.csv --output-json params.json
    CLI->>CW: run_heston_calibration(args)
    CW->>Loader: load_quotes_csv(path)
    Loader-->>CW: List[OptionQuote]
    CW->>CalHeston: calibrate_heston_to_quotes(quotes) [L-BFGS-B]
    CalHeston->>HestonVanilla: heston_vanilla_price_mc() [per iteration]
    HestonVanilla->>SimHeston: simulate_heston_paths()
    CalHeston-->>CW: kappa, theta, sigma_v, rho, v0 + RMSE
    CW-->>User: params.json + RMSE report

    User->>CLI: barrier-price --model heston --params-json params.json
    CLI->>CW: run_barrier(args)
    CW->>Exotics: price_barrier_heston_mc(params)
    Exotics->>SimHeston: simulate_heston_paths()
    Exotics-->>User: Barrier option price + std error
```

---

## Pricing Commands

```mermaid
graph LR
    subgraph PricingWorkflow
        rbs["run_bs()"] --> bse["bs_price_and_greeks\nimplied_volatility\nplot_greeks"]
        rmc["run_mc()"] --> gbm["simulate_gbm_paths"]
        rhp["run_heston()"] --> hmc["heston_vanilla_price_mc\nsimulate_heston_paths"]
        rbi["run_binomial()"] --> bin["binomial_price_and_greeks\nplot_greeks"]
        rba["run_barrier()"] --> bar["price_barrier_mc\nprice_barrier_heston_mc"]
        rlb["run_lookback()"] --> lbk["price_lookback_mc\nprice_lookback_heston_mc"]
    end
```

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

Prices European options under the Heston model via Monte Carlo. Accepts either explicit Heston parameters or a pre-calibrated JSON file.

**Heston model dynamics:**

```
dS = r·S·dt + √V·S·dW₁
dV = κ(θ - V)dt + σᵥ·√V·dW₂
corr(dW₁, dW₂) = ρ
```

**Parameters:** κ (mean reversion speed), θ (long-run variance), σᵥ (vol of vol), ρ (correlation), v₀ (initial variance).

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

Prices path-dependent barrier options (Up/Down, In/Out). GBM uses Brownian bridge correction for continuous barrier monitoring; Heston uses discrete path monitoring.

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

```mermaid
graph LR
    subgraph CalibrationWorkflow
        run["run()"] --> lq["load_quotes_csv"] & na["check_no_arbitrage\ncheck_put_call_parity"] & sm["calibrate_surface_with_smoothing"] & viz["plot_implied_vol_surface\nplot_implied_vol_smile"]
        rhc["run_heston_calibration()"] --> lq & ch["calibrate_heston_to_quotes\n→ heston_vanilla_price_mc\n→ simulate_heston_paths"] & sj["save_heston_params_json"]
    end
```

### `calibrate-surface` — Implied Volatility Surface

Calibrates an implied volatility surface from market quotes. Performs static no-arbitrage checks (butterfly and calendar spread violations) and produces a smoothed volatility grid.

**Arguments:** `--input-csv`, `--S0`, `--r`.

**Workflow:** `CalibrationWorkflow.run()`

```bash
derivatives-pricing-model calibrate-surface --input-csv quotes.csv --S0 100 --r 0.05
```

---

### `calibrate-heston` — Heston Parameter Fitting

Calibrates the five Heston parameters (κ, θ, σᵥ, ρ, v₀) to vanilla option quotes using **L-BFGS-B** optimization with a soft Feller condition penalty (`2κθ ≥ σᵥ²`). Outputs RMSE on prices and implied volatilities.

**Arguments:** `--input-csv`, `--S0`, `--r` + calibration args + initial guesses (`--init-kappa`, `--init-theta`, `--init-sigma-v`, `--init-rho`, `--init-v0`).

**Workflow:** `CalibrationWorkflow.run_heston_calibration()`

```bash
derivatives-pricing-model calibrate-heston --input-csv examples/heston/synthetic_heston_quotes.csv --S0 100 --r 0.03 --M 4000 --n-steps 64 --maxiter 30 --antithetic --output-json examples/heston/calibrated_params.json
```

---

## Hedging & Risk Commands

```mermaid
graph LR
    subgraph HedgingWorkflow
        rh["run()"] --> gbm["simulate_gbm_paths\nsimulate_heston_paths"] & dh["simulate_discrete_hedging\n→ bs_price_and_greeks"] & ves["calculate_var_es"] & pnl["plot_pnl_distribution\nplot_pnl_surface\nplot_pnl_comparison"]
    end

    subgraph StressWorkflow
        rs["run()"] --> gbm2["simulate_gbm/heston_paths"] & stu["generate_student_t_paths\napply_spot_vol_shock"] & dh2["simulate_discrete_hedging"] & ves2["calculate_var_es"]
    end

    subgraph RiskWorkflow
        rr["run()"] --> op["optimize_portfolio\nSLSQP solver"]
    end
```

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

**Scenarios:**
- **Gaussian**: standard GBM simulation
- **Student-t** (`--df`): fat-tailed returns via `generate_student_t_paths()`
- **Spot/Vol shock** (`--spot-shock`, `--vol-shock`): instantaneous parallel shifts via `apply_spot_vol_shock()`

**Arguments:** Standard market args + `--M`, `--n-steps`, `--df`, `--spot-shock`, `--vol-shock`, `--seed`.

**Workflow:** `StressWorkflow.run()`

```bash
derivatives-pricing-model stress-run --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --n-steps 252 --M 1000 --df 4.0 --spot-shock -0.10 --vol-shock 0.05 --seed 42
```

---

### `optimize-risk` — Portfolio Risk Optimization

Optimizes portfolio hedging under linear Greek neutrality constraints and quadratic residual-risk penalties using **SLSQP**.

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

## Quant-Engine Commands (v0.2)

The commands below use the new high-precision engines added in the quant upgrade.  They call engines directly (no heavyweight workflow wrapper needed) and all support `--output-json`.

---

### `fourier-price` — Heston Fourier Pricing (COS / Carr-Madan / Lewis)

Prices a Heston European option via one or all three Fourier methods:
- **COS** (Fang-Oosterlee 2008): N=256 cosine series, cumulant-based truncation. Fast default for calibration loops (~0.1 ms per strike strip).
- **Carr-Madan** (1999): FFT with α-damping and Simpson weights; prices a full log-strike strip in one transform.
- **Lewis** (2001): single `scipy.quad` integral; simplest independent cross-check.

All three are mutually consistent to < 2 cents under standard parameters.

**Arguments:** Standard market args (no `--sigma`) + Heston params + `--method {cos,carr-madan,lewis,all}`.

```bash
python main.py fourier-price --S0 100 --K 100 --T 1 --r 0.05 \
  --kappa 2.0 --theta 0.04 --sigma-v 0.3 --rho -0.7 --v0 0.04 \
  --method all

python main.py fourier-price --S0 100 --K 100 --T 1 --r 0.05 \
  --kappa 2.0 --theta 0.04 --sigma-v 0.3 --rho -0.7 --v0 0.04 \
  --method cos --put --output-json heston_put.json
```

---

### `american-price` — Longstaff-Schwartz American MC

Prices an American option using Longstaff-Schwartz (2001) backward-induction Monte Carlo.
Returns price, ±95 % confidence interval, standard error, early-exercise fraction, and the estimated exercise boundary.

**Key property:** the LSM exercise strategy is sub-optimal → price is a *lower bound* on the true American price.

**Arguments:** Standard market args + `--n-paths` (default 10000) + `--n-steps` (default 100) + `--poly-degree` (default 3) + `--seed`.

```bash
python main.py american-price --S0 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --put \
  --n-paths 50000 --n-steps 200 --seed 42

# Cross-check: American call without dividends ≈ European call
python main.py american-price --S0 100 --K 100 --T 1 --r 0.05 --sigma 0.2 \
  --n-paths 20000 --seed 0
```

---

### `merton-price` — Merton Jump-Diffusion

Prices a European option under the Merton (1976) jump-diffusion model via Poisson-weighted Black-Scholes series (machine-precise for λT ≤ 20, truncated at `--n-terms`).

**Arguments:** Standard market args + `--lam` (jump intensity) + `--mu-J` (mean log-jump) + `--sigma-J` (log-jump std) + `--n-terms`.

```bash
python main.py merton-price --S0 100 --K 100 --T 1 --r 0.05 --sigma 0.2 \
  --lam 1.0 --mu-J -0.10 --sigma-J 0.15

# With λ=0 this reduces to Black-Scholes
python main.py merton-price --S0 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --lam 0
```

---

### `heston-pde-price` — Heston 2-D ADI PDE

Prices a Heston European option by solving the 2-D Heston PDE on a (log-S, v) grid via Douglas-Rachford ADI splitting.  Uses a sinh-based non-uniform v-grid concentrated near v = 0.

Cross-validates against `fourier-price --method cos` to within 1.5 % ATM.

**Arguments:** Standard market args (no `--sigma`) + Heston params + `--Nx` + `--Nv` + `--Nt`.

```bash
python main.py heston-pde-price --S0 100 --K 100 --T 1 --r 0.05 \
  --kappa 2.0 --theta 0.04 --sigma-v 0.3 --rho -0.7 --v0 0.04 \
  --Nx 80 --Nv 40 --Nt 80
```

---

### `sabr-vol` — SABR Implied Volatility

Computes lognormal (Black) or normal (Bachelier) SABR implied volatility at a single strike via the Hagan (2002) formula.

**Arguments:** `--F` (forward) + `--K` + `--T` + `--alpha` + `--beta` (default 0.5) + `--rho` + `--nu` + `--correction {hagan,obloj}` + `--normal`.

```bash
python main.py sabr-vol --F 100 --K 95 --T 1 --alpha 2.0 --beta 0.5 --rho -0.3 --nu 0.4

# Normal SABR (for negative-rate environments)
python main.py sabr-vol --F 0.01 --K 0.005 --T 2 --alpha 0.005 --rho -0.2 --nu 0.3 --normal
```

*Note:* With `beta=0.5` and `F=100`, `alpha` has units vol × F^(β−1) = vol × F^(−0.5).  To get ≈20% vol at F=100 with beta=0.5, use `alpha≈2.0`.

---

### `sabr-calibrate` — Calibrate SABR Smile

Fits SABR parameters (α, ρ, ν) to a market smile with β fixed.  α is determined analytically from the ATM cubic at each (ρ, ν) candidate; ρ and ν are optimised via L-BFGS-B.

**Arguments:** `--F` + `--T` + `--strikes '...'` (comma-separated) + `--vols '...'` (comma-separated implied vols) + `--beta`.

```bash
python main.py sabr-calibrate --F 100 --T 1 \
  --strikes "90,95,100,105,110" \
  --vols   "0.22,0.21,0.20,0.21,0.22" \
  --beta 0.5 --output-json sabr_params.json
```

---

### `dupire-surface` — Dupire Local Volatility

Computes Dupire (1994) local volatility σ_loc(K, T) from an implied-vol surface using central finite differences in the Derman-Kani total-variance form.

The current implementation takes a flat surface (constant σ); for a non-flat surface, call `dupire_local_vol` programmatically with a custom callable.

**Arguments:** `--S0` + `--K` + `--T` + `--r` + `--sigma` (flat implied vol surface value).

```bash
python main.py dupire-surface --S0 100 --K 100 --T 1 --r 0.05 --sigma 0.2
# For a flat surface: local vol ≈ implied vol = 20%

python main.py dupire-surface --S0 100 --K 90 --T 0.5 --r 0.05 --sigma 0.25 \
  --output-json local_vol.json
```

---

## Fast Calibration Pipeline (v0.2)

With COS-powered calibration, the entire Heston calibration loop runs in seconds instead of minutes:

```bash
# Step 1: Calibrate with COS (default, ~1000× faster than MC)
python main.py calibrate-heston \
  --input-csv examples/heston/synthetic_heston_quotes.csv \
  --S0 100 --r 0.05 --maxiter 300 --output-json params.json

# Step 2: Verify with Fourier cross-check
python main.py fourier-price --S0 100 --K 100 --T 0.5 --r 0.05 \
  --kappa 2.0 --theta 0.04 --sigma-v 0.3 --rho -0.7 --v0 0.04 \
  --method all

# Step 3: American option from calibrated vol
python main.py american-price --S0 100 --K 100 --T 1 --r 0.05 --sigma 0.2 \
  --put --n-paths 50000 --seed 42
```

---

## Visualization

All commands that generate plots support two additional flags:

| Flag | Description |
| --- | --- |
| `--save-plots <directory>` | Save all plots as PNG files at 300 DPI in the specified directory |
| `--no-plots` | Suppress all graphical output |

### Available Visualizations

| Command | Plots Generated |
| --- | --- |
| `bs-price` | Greek sensitivity curves (Delta, Gamma, Vega, Theta, Rho, Vanna vs spot), payoff diagram with breakeven |
| `binomial-price` | Greek sensitivity curves, payoff diagram with breakeven |
| `mc-price` | Monte Carlo convergence plot with 95% CI band and BS analytical benchmark |
| `calibrate-surface` | 3D implied volatility surface (cubic interpolation), volatility smile by moneyness per maturity |
| `hedge-sim` | P&L distribution with VaR tail shading, P&L attribution waterfall, hedging path fan chart, P&L vs final spot (binned with CI band), P&L density hexbin, 3D P&L surface, baseline vs stress comparison |
| `stress-run` | Scenario comparison (VaR/ES grouped bar chart across Gaussian, Student-t, spot/vol shock, short convexity) |
| `hist-vol` | Rolling realized volatility with underlying price overlay and high-vol regime shading |

A consolidated multi-page PDF of all plots (using synthetic seeded data) can be generated with:

```bash
PYTHONPATH=src python scripts/generate_visualizations_report.py
# Output: output/visualizations.pdf
```

Example usage of CLI flags:

```bash
derivatives-pricing-model hedge-sim --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --n-steps 252 --M 1000 --cost 0.001 --save-plots output/plots
derivatives-pricing-model hedge-sim --S0 100 --K 100 --T 1.0 --r 0.05 --sigma 0.2 --n-steps 252 --M 1000 --cost 0.001 --no-plots
```

---

## End-to-End Pipeline: Calibration → Exotic Pricing

```mermaid
graph TD
    A["Market option quotes\nquotes.csv"] --> B["calibrate-heston\nL-BFGS-B optimizer\n+ Feller penalty"]
    B --> C["params.json\n(κ, θ, σᵥ, ρ, v₀)"]
    C --> D["heston-price\nvanilla pricing check"]
    C --> E["barrier-price --model heston\npath-dependent barrier"]
    C --> F["lookback-price --model heston\nfloating / fixed strike"]
    C --> G["hedge-sim --model heston\ndelta hedging simulation"]
```

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

---

## Argument Parsing Infrastructure

The CLI uses shared utility functions in `src/cli/parser.py` to inject standard argument groups:

| Utility Function | Purpose | Key Arguments |
| --- | --- | --- |
| `add_standard_market_args` | Standard Black-Scholes inputs | `--S0`, `--K`, `--T`, `--r`, `--sigma`, `--put` |
| `add_heston_args` | Parameters for Heston MC simulation | `--kappa`, `--theta`, `--sigma-v`, `--rho`, `--v0`, `--params-json`, `--antithetic` |
| `add_heston_calibration_args` | Settings for the Heston optimizer | `--M`, `--n-steps`, `--maxiter`, `--weight-mode`, `--output-json` |

---

## Module Reference

### CLI Layer

| Module | Path | Purpose |
|---|---|---|
| `main` | `cli/main.py` | Command dispatcher |
| `parser` | `cli/parser.py` | Shared argument groups |
| `bs_price` | `cli/commands/bs_price.py` | Black-Scholes command |
| `mc_price` | `cli/commands/mc_price.py` | GBM Monte Carlo command |
| `heston_price` | `cli/commands/heston_price.py` | Heston pricing command |
| `binomial_price` | `cli/commands/binomial_price.py` | Binomial tree command |
| `barrier_price` | `cli/commands/barrier_price.py` | Barrier option command |
| `lookback_price` | `cli/commands/lookback_price.py` | Lookback option command |
| `calibrate_surface` | `cli/commands/calibrate_surface.py` | IV surface calibration command |
| `calibrate_heston` | `cli/commands/calibrate_heston.py` | Heston calibration command (COS default) |
| `hedge_sim` | `cli/commands/hedge_sim.py` | Hedging simulation command |
| `stress_run` | `cli/commands/stress_run.py` | Stress testing command |
| `optimize_risk` | `cli/commands/optimize_risk.py` | Portfolio optimization command |
| `hist_vol` | `cli/commands/hist_vol.py` | Historical volatility command |
| `fourier_price` | `cli/commands/fourier_price.py` | Heston Fourier pricing (COS/CM/Lewis) ★ |
| `american_price` | `cli/commands/american_price.py` | Longstaff-Schwartz American MC ★ |
| `merton_price` | `cli/commands/merton_price.py` | Merton jump-diffusion series ★ |
| `pde_heston_price` | `cli/commands/pde_heston_price.py` | Heston 2-D ADI PDE ★ |
| `sabr_vol` | `cli/commands/sabr_vol.py` | SABR implied vol (Hagan 2002) ★ |
| `sabr_calibrate` | `cli/commands/sabr_calibrate.py` | SABR smile calibration ★ |
| `dupire_surface` | `cli/commands/dupire_surface.py` | Dupire local vol at (K, T) ★ |

### Workflows Layer

| Class | Path | Methods |
|---|---|---|
| `PricingWorkflow` | `workflows/pricing_workflow.py` | `run_bs`, `run_mc`, `run_heston`, `run_binomial`, `run_barrier`, `run_lookback` |
| `CalibrationWorkflow` | `workflows/calibration_workflow.py` | `run`, `run_heston_calibration` |
| `HedgingWorkflow` | `workflows/hedging_workflow.py` | `run` |
| `StressWorkflow` | `workflows/stress_workflow.py` | `run` |
| `RiskWorkflow` | `workflows/risk_workflow.py` | `run` |
| `HistoricalVolWorkflow` | `workflows/historical_vol_workflow.py` | `run` |

### Engines Layer

| Module | Path | Public Functions |
|---|---|---|
| `black_scholes` | `engines/pricing/black_scholes.py` | `bs_price_and_greeks` |
| `binomial` | `engines/pricing/binomial.py` | `binomial_price`, `binomial_price_and_greeks` |
| `heston_vanilla` | `engines/pricing/heston_vanilla.py` | `heston_vanilla_price_mc` |
| `exotics` | `engines/pricing/exotics.py` | `price_barrier_mc`, `price_barrier_heston_mc`, `price_lookback_mc`, `price_lookback_heston_mc` |
| `implied_vol` | `engines/pricing/implied_vol.py` | `implied_volatility` |
| `heston_fourier` ★ | `engines/pricing/heston_fourier.py` | `heston_characteristic_function`, `heston_price_carr_madan`, `heston_price_cos`, `heston_price_lewis` |
| `sabr` ★ | `engines/pricing/sabr.py` | `sabr_implied_vol`, `sabr_normal_vol`, `calibrate_sabr` |
| `pde` ★ | `engines/pricing/pde.py` | `bs_pde_price`, `heston_pde_price` |
| `american_mc` ★ | `engines/pricing/american_mc.py` | `american_option_lsm` |
| `jump_diffusion` ★ | `engines/pricing/jump_diffusion.py` | `merton_characteristic_function`, `merton_price`, `simulate_merton_paths` |
| `local_vol` ★ | `engines/pricing/local_vol.py` | `dupire_local_vol`, `calibrate_svi`, `check_svi_arbitrage` |
| `gbm` | `engines/simulation/gbm.py` | `simulate_gbm_paths`, `simulate_gbm_paths_student_t` |
| `heston` (sim) | `engines/simulation/heston.py` | `simulate_heston_paths`, `simulate_heston_paths_qe`, `simulate_heston_paths_euler`, `check_feller_condition` |
| `variance_reduction` ★ | `engines/simulation/variance_reduction.py` | `mc_with_control_variate`, `sobol_standard_normal`, `apply_moment_matching` |
| `jit` ★ | `engines/simulation/jit.py` | `maybe_jit`, `gbm_step_loop`, `heston_euler_step_loop` |
| `surface` | `engines/calibration/surface.py` | `check_no_arbitrage`, `check_put_call_parity`, `calibrate_surface_with_smoothing` |
| `heston` (cal) | `engines/calibration/heston.py` | `calibrate_heston_to_quotes` (default `pricing_method="cos"`) |
| `discrete_hedging` | `engines/hedging/discrete_hedging.py` | `simulate_discrete_hedging` |
| `scenario` | `engines/stress/scenario.py` | `calculate_var_es`, `generate_student_t_paths`, `apply_spot_vol_shock`, `generate_short_convexity_scenario` |
| `optimization` | `engines/risk/optimization.py` | `optimize_portfolio` |

*★ = added in v0.2 quant-engine upgrade*

### Support Layer

| Module | Path | Purpose |
|---|---|---|
| `loaders` | `io/loaders.py` | `load_quotes_csv`, `load_historical_prices` |
| `plots` | `visualization/plots.py` | All `plot_*` functions |
| `validation` | `utils/validation.py` | Input validation helpers |
| `heston_params` | `utils/heston_params.py` | `load_heston_params_json`, `save_heston_params_json` |
| `logging_config` | `utils/logging_config.py` | Logger setup |
| `domain` | `models/domain.py` | `OptionQuote` dataclass |
