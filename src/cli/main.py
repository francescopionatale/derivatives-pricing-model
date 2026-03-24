import argparse
from quant_derivatives.cli.commands import (
    barrier_price,
    binomial_price,
    bs_price,
    calibrate_heston,
    calibrate_surface,
    hedge_sim,
    heston_price,
    hist_vol,
    lookback_price,
    mc_price,
    optimize_risk,
    stress_run,
)


def main():
    parser = argparse.ArgumentParser(description="Quant Derivatives CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    binomial_price.setup_parser(subparsers)
    bs_price.setup_parser(subparsers)
    mc_price.setup_parser(subparsers)
    hedge_sim.setup_parser(subparsers)
    stress_run.setup_parser(subparsers)
    calibrate_surface.setup_parser(subparsers)
    calibrate_heston.setup_parser(subparsers)
    hist_vol.setup_parser(subparsers)
    heston_price.setup_parser(subparsers)
    barrier_price.setup_parser(subparsers)
    lookback_price.setup_parser(subparsers)
    optimize_risk.setup_parser(subparsers)

    args = parser.parse_args()

    if args.command == "binomial-price":
        binomial_price.run(args)
    elif args.command == "bs-price":
        bs_price.run(args)
    elif args.command == "mc-price":
        mc_price.run(args)
    elif args.command == "hedge-sim":
        hedge_sim.run(args)
    elif args.command == "stress-run":
        stress_run.run(args)
    elif args.command == "calibrate-surface":
        calibrate_surface.run(args)
    elif args.command == "calibrate-heston":
        calibrate_heston.run(args)
    elif args.command == "hist-vol":
        hist_vol.run(args)
    elif args.command == "heston-price":
        heston_price.run(args)
    elif args.command == "barrier-price":
        barrier_price.run(args)
    elif args.command == "lookback-price":
        lookback_price.run(args)
    elif args.command == "optimize-risk":
        optimize_risk.run(args)

    return 0
