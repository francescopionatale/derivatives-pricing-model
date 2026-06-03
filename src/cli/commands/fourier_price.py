"""CLI: Heston Fourier pricing (COS / Carr-Madan / Lewis)."""
from __future__ import annotations

import json

from cli.parser import add_heston_args, add_standard_market_args


def setup_parser(subparsers):
    p = subparsers.add_parser("fourier-price", help="Heston Fourier pricing (COS, Carr-Madan, Lewis)")
    add_standard_market_args(p, require_sigma=False)
    add_heston_args(p, include_model=False)
    p.add_argument("--method", choices=["cos", "carr-madan", "lewis", "all"], default="cos",
                   help="Fourier method (default: cos)")
    p.add_argument("--output-json", type=str, default=None, help="Write result JSON to this file")


def run(args):
    from engines.pricing.heston_fourier import (
        heston_price_carr_madan,
        heston_price_cos,
        heston_price_lewis,
    )

    kw = dict(
        S0=args.S0, K=args.K, T=args.T, r=args.r,
        kappa=args.kappa, theta=args.theta or 0.04,
        sigma_v=args.sigma_v, rho=args.rho,
        v0=args.v0 or (args.sigma ** 2 if args.sigma else 0.04),
        is_call=args.is_call,
    )

    results = {}
    if args.method in ("cos", "all"):
        results["cos"] = heston_price_cos(**kw)
    if args.method in ("carr-madan", "all"):
        results["carr-madan"] = heston_price_carr_madan(**kw)
    if args.method in ("lewis", "all"):
        results["lewis"] = heston_price_lewis(**kw)

    for method, res in results.items():
        print(f"[{method}] price = {res['price']:.6f}")

    if args.output_json:
        out = {m: {k: (float(v) if hasattr(v, "__float__") else v) for k, v in r.items() if k != "strikes"}
               for m, r in results.items()}
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Result written to {args.output_json}")
