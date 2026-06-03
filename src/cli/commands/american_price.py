"""CLI: Longstaff-Schwartz American option pricing."""
from __future__ import annotations

from cli.parser import add_standard_market_args


def setup_parser(subparsers):
    p = subparsers.add_parser("american-price", help="American option via Longstaff-Schwartz LSM")
    add_standard_market_args(p)
    p.add_argument("--n-paths", type=int, default=10000, help="Number of MC paths")
    p.add_argument("--n-steps", type=int, default=100, help="Time steps per path")
    p.add_argument("--poly-degree", type=int, default=3, help="Polynomial degree for continuation regression")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--output-json", type=str, default=None, help="Write result JSON to this file")


def run(args):
    import json as _json

    from engines.pricing.american_mc import american_option_lsm

    res = american_option_lsm(
        S0=args.S0, K=args.K, T=args.T, r=args.r, sigma=args.sigma,
        is_call=args.is_call, n_paths=args.n_paths, n_steps=args.n_steps,
        poly_degree=args.poly_degree, seed=args.seed,
    )

    print(f"[lsm] price = {res['price']:.6f}  std_err = {res['std_err']:.6f}")
    print(f"      95% CI = [{res['ci_low']:.4f}, {res['ci_hi']:.4f}]")
    print(f"      exercise_frac = {res['exercise_frac']:.4f}")

    if args.output_json:
        out = {k: (float(v) if hasattr(v, "__float__") else v)
               for k, v in res.items() if k not in ("exercise_boundary",)}
        with open(args.output_json, "w") as f:
            _json.dump(out, f, indent=2)
        print(f"Result written to {args.output_json}")
