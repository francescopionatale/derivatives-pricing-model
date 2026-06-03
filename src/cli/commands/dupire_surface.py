"""CLI: Dupire local vol at a single (K, T) point given a flat or parametric surface."""
from __future__ import annotations

import json


def setup_parser(subparsers):
    p = subparsers.add_parser("dupire-surface",
                              help="Dupire local vol at (K, T) from a flat implied-vol surface")
    p.add_argument("--S0", type=float, required=True, help="Current spot price")
    p.add_argument("--K", type=float, required=True, help="Strike")
    p.add_argument("--T", type=float, required=True, help="Maturity (years)")
    p.add_argument("--r", type=float, required=True, help="Risk-free rate")
    p.add_argument("--sigma", type=float, required=True,
                   help="Flat implied vol (used as constant surface σ(K,T)=sigma)")
    p.add_argument("--output-json", type=str, default=None, help="Write result JSON here")


def run(args):
    from engines.pricing.local_vol import dupire_local_vol

    def flat_surface(K: float, T: float) -> float:  # noqa: ARG001
        return args.sigma

    res = dupire_local_vol(K=args.K, T=args.T, S0=args.S0, r=args.r, iv_surface=flat_surface)

    print(f"[dupire] σ_loc({args.K:.1f}, {args.T:.2f}) = {res['local_vol']:.6f}  "
          f"({res['local_vol']*100:.4f}%)")

    if args.output_json:
        out = {k: float(v) if hasattr(v, "__float__") else v for k, v in res.items()}
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Result written to {args.output_json}")
