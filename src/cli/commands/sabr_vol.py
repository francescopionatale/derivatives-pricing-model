"""CLI: SABR implied volatility (single strike or smile strip)."""
from __future__ import annotations

import json


def setup_parser(subparsers):
    p = subparsers.add_parser("sabr-vol", help="SABR (Hagan 2002) implied vol at a given strike")
    p.add_argument("--F", type=float, required=True, help="Forward price")
    p.add_argument("--K", type=float, required=True, help="Strike price")
    p.add_argument("--T", type=float, required=True, help="Time to maturity (years)")
    p.add_argument("--alpha", type=float, required=True, help="SABR alpha (initial vol level)")
    p.add_argument("--beta", type=float, default=0.5, help="CEV exponent (default 0.5)")
    p.add_argument("--rho", type=float, required=True, help="Spot-vol correlation")
    p.add_argument("--nu", type=float, required=True, help="Vol-of-vol")
    p.add_argument("--correction", choices=["hagan", "obloj"], default="hagan",
                   help="Formula variant (default: hagan)")
    p.add_argument("--normal", action="store_true",
                   help="Compute normal (Bachelier) SABR vol instead of lognormal")
    p.add_argument("--output-json", type=str, default=None, help="Write result JSON to this file")


def run(args):
    from engines.pricing.sabr import sabr_implied_vol, sabr_normal_vol

    if args.normal:
        res = sabr_normal_vol(F=args.F, K=args.K, T=args.T, alpha=args.alpha, rho=args.rho, nu=args.nu)
        label = "normal SABR"
    else:
        res = sabr_implied_vol(F=args.F, K=args.K, T=args.T, alpha=args.alpha, beta=args.beta,
                               rho=args.rho, nu=args.nu, correction=args.correction)
        label = f"SABR ({args.correction})"

    print(f"[{label}] σ = {res['sigma']:.6f}  ({res['sigma']*100:.4f}%)")

    if args.output_json:
        out = {k: float(v) if hasattr(v, "__float__") else v for k, v in res.items()}
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Result written to {args.output_json}")
