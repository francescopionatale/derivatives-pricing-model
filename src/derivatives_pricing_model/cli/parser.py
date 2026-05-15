"""Shared argparse utilities."""
import argparse


def add_standard_market_args(parser: argparse.ArgumentParser, *, require_sigma: bool = True):
    parser.add_argument("--S0", type=float, required=True, help="Spot price")
    parser.add_argument("--K", type=float, required=True, help="Strike price")
    parser.add_argument("--T", type=float, required=True, help="Time to maturity (years)")
    parser.add_argument("--r", type=float, required=True, help="Risk-free rate (decimal)")
    parser.add_argument("--sigma", type=float, required=require_sigma, default=None, help="Volatility (decimal)")
    parser.add_argument("--put", dest="is_call", action="store_false", help="Priced as Put")
    parser.set_defaults(is_call=True)


def add_heston_args(parser: argparse.ArgumentParser, *, include_model: bool = True, default_model: str = "gbm"):
    if include_model:
        parser.add_argument("--model", choices=["gbm", "heston"], default=default_model, help="Simulation model for path generation")
    parser.add_argument("--params-json", type=str, default=None, help="Path to a JSON file containing calibrated Heston parameters")
    parser.add_argument("--kappa", type=float, default=2.0, help="Heston variance mean reversion speed")
    parser.add_argument("--theta", type=float, default=None, help="Heston long-run variance; defaults to sigma^2 or calibrated theta")
    parser.add_argument("--sigma-v", dest="sigma_v", type=float, default=0.3, help="Heston volatility of variance")
    parser.add_argument("--rho", type=float, default=-0.7, help="Heston spot/variance correlation")
    parser.add_argument("--v0", type=float, default=None, help="Heston initial variance; defaults to sigma^2 or calibrated v0")
    parser.add_argument("--antithetic", action="store_true", help="Use antithetic variates when supported")


def add_heston_calibration_args(parser: argparse.ArgumentParser):
    parser.add_argument("--M", type=int, default=4000, help="Number of Monte Carlo paths used inside calibration")
    parser.add_argument("--n-steps", type=int, default=64, help="Time steps per path used inside calibration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for common random numbers")
    parser.add_argument("--maxiter", type=int, default=30, help="Maximum optimizer iterations")
    parser.add_argument("--weight-mode", choices=["spread", "uniform"], default="spread", help="Quote weighting scheme during calibration")
    parser.add_argument("--antithetic", action="store_true", help="Use antithetic variates during calibration")
    parser.add_argument("--init-kappa", type=float, default=2.0, help="Initial guess for kappa")
    parser.add_argument("--init-theta", type=float, default=0.04, help="Initial guess for theta")
    parser.add_argument("--init-sigma-v", dest="init_sigma_v", type=float, default=0.30, help="Initial guess for sigma_v")
    parser.add_argument("--init-rho", type=float, default=-0.70, help="Initial guess for rho")
    parser.add_argument("--init-v0", type=float, default=0.04, help="Initial guess for v0")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path where calibrated Heston parameters will be written")
