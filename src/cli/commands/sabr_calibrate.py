"""CLI: Calibrate SABR (alpha, rho, nu) to a market smile."""
from __future__ import annotations

import json


def setup_parser(subparsers):
    p = subparsers.add_parser("sabr-calibrate", help="Calibrate SABR to a smile (comma-separated strikes/vols)")
    p.add_argument("--F", type=float, required=True, help="Forward price")
    p.add_argument("--T", type=float, required=True, help="Time to maturity (years)")
    p.add_argument("--strikes", type=str, required=True,
                   help="Comma-separated strikes, e.g. 90,95,100,105,110")
    p.add_argument("--vols", type=str, required=True,
                   help="Comma-separated market lognormal implied vols, e.g. 0.22,0.21,0.20,0.21,0.22")
    p.add_argument("--beta", type=float, default=0.5, help="CEV exponent, fixed (default 0.5)")
    p.add_argument("--output-json", type=str, default=None, help="Write calibrated params JSON here")


def run(args):
    import numpy as np

    from engines.pricing.sabr import calibrate_sabr

    strikes = np.array([float(x) for x in args.strikes.split(",")])
    vols = np.array([float(x) for x in args.vols.split(",")])

    result = calibrate_sabr(F=args.F, T=args.T, strikes=strikes, market_vols=vols, beta=args.beta)

    print(f"[sabr-calibrate] alpha={result['alpha']:.6f}  rho={result['rho']:.4f}  "
          f"nu={result['nu']:.4f}  (beta={result['beta']:.2f} fixed)")
    print(f"  rmse_vol={result['rmse']:.6f}")

    if args.output_json:
        out = {
            "alpha": float(result["alpha"]),
            "beta": float(result["beta"]),
            "rho": float(result["rho"]),
            "nu": float(result["nu"]),
            "rmse_vol": float(result["rmse"]),
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Result written to {args.output_json}")
