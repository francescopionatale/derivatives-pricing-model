from __future__ import annotations

import argparse

from cli.commands import (
    american_price,
    barrier_price,
    binomial_price,
    bs_price,
    calibrate_heston,
    calibrate_surface,
    dupire_surface,
    fourier_price,
    hedge_sim,
    heston_price,
    hist_vol,
    lookback_price,
    mc_price,
    merton_price,
    optimize_risk,
    pde_heston_price,
    sabr_calibrate,
    sabr_vol,
    stress_run,
)


def main():
    parser = argparse.ArgumentParser(description="Derivatives Pricing Model CLI")
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
    # Wave-2 engines
    fourier_price.setup_parser(subparsers)
    american_price.setup_parser(subparsers)
    merton_price.setup_parser(subparsers)
    pde_heston_price.setup_parser(subparsers)
    sabr_vol.setup_parser(subparsers)
    sabr_calibrate.setup_parser(subparsers)
    dupire_surface.setup_parser(subparsers)

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
    elif args.command == "fourier-price":
        fourier_price.run(args)
    elif args.command == "american-price":
        american_price.run(args)
    elif args.command == "merton-price":
        merton_price.run(args)
    elif args.command == "heston-pde-price":
        pde_heston_price.run(args)
    elif args.command == "sabr-vol":
        sabr_vol.run(args)
    elif args.command == "sabr-calibrate":
        sabr_calibrate.run(args)
    elif args.command == "dupire-surface":
        dupire_surface.run(args)

    return 0
