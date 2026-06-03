"""CLI: Merton jump-diffusion pricing."""
from __future__ import annotations

from cli.parser import add_standard_market_args


def setup_parser(subparsers):
    p = subparsers.add_parser("merton-price", help="Merton (1976) jump-diffusion option pricing")
    add_standard_market_args(p)
    p.add_argument("--lam", type=float, default=1.0, help="Jump intensity (jumps/year)")
    p.add_argument("--mu-J", type=float, default=-0.1, dest="mu_J", help="Mean log-jump size")
    p.add_argument("--sigma-J", type=float, default=0.15, dest="sigma_J", help="Std of log-jump size")
    p.add_argument("--n-terms", type=int, default=50, help="Series truncation terms")
    p.add_argument("--output-json", type=str, default=None, help="Write result JSON to this file")


def run(args):
    import json as _json

    from engines.pricing.jump_diffusion import merton_price

    res = merton_price(
        S0=args.S0, K=args.K, T=args.T, r=args.r, sigma=args.sigma,
        lam=args.lam, mu_J=args.mu_J, sigma_J=args.sigma_J,
        is_call=args.is_call, n_terms=args.n_terms,
    )

    print(f"[merton-series] price = {res['price']:.6f}  kbar = {res['kbar']:.6f}")
    print(f"                λ' = {res['lam_prime']:.4f}  n_terms_used = {res['n_terms']}")

    if args.output_json:
        out = {k: float(v) if hasattr(v, "__float__") else v for k, v in res.items()}
        with open(args.output_json, "w") as f:
            _json.dump(out, f, indent=2)
        print(f"Result written to {args.output_json}")
