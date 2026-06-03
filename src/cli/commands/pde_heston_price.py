"""CLI: Heston 2-D ADI PDE pricing."""
from __future__ import annotations

from cli.parser import add_heston_args, add_standard_market_args


def setup_parser(subparsers):
    p = subparsers.add_parser("heston-pde-price", help="Heston European option via 2-D Douglas-Rachford ADI PDE")
    add_standard_market_args(p, require_sigma=False)
    add_heston_args(p, include_model=False)
    p.add_argument("--Nx", type=int, default=80, dest="N_x", help="Log-spot grid nodes")
    p.add_argument("--Nv", type=int, default=40, dest="N_v", help="Variance grid nodes")
    p.add_argument("--Nt", type=int, default=80, dest="N_t", help="Time steps")
    p.add_argument("--output-json", type=str, default=None, help="Write result JSON to this file")


def run(args):
    import json

    from engines.pricing.pde import heston_pde_price

    res = heston_pde_price(
        S0=args.S0, K=args.K, T=args.T, r=args.r,
        kappa=args.kappa, theta=args.theta or 0.04,
        sigma_v=args.sigma_v, rho=args.rho,
        v0=args.v0 or (args.sigma ** 2 if args.sigma else 0.04),
        is_call=args.is_call,
        N_x=args.N_x, N_v=args.N_v, N_t=args.N_t,
    )

    print(f"[heston-pde-adi] price = {res['price']:.6f}")

    if args.output_json:
        out = {k: float(v) if hasattr(v, "__float__") else v
               for k, v in res.items() if k not in ("grid_S", "grid_v", "grid_V")}
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Result written to {args.output_json}")
