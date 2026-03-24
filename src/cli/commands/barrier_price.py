from quant_derivatives.cli.parser import add_standard_market_args, add_heston_args
from quant_derivatives.workflows.pricing_workflow import PricingWorkflow



def setup_parser(subparsers):
    parser = subparsers.add_parser("barrier-price", help="Barrier option pricing (GBM or Heston Monte Carlo)")
    add_standard_market_args(parser, require_sigma=False)
    parser.add_argument("--barrier", type=float, required=True, help="Barrier level")
    parser.add_argument("--direction", choices=["up", "down"], default="up", help="Barrier direction")
    parser.add_argument("--barrier-style", choices=["in", "out"], default="out", help="Knock-in or knock-out")
    parser.add_argument("--M", type=int, default=10000, help="Number of paths")
    parser.add_argument("--n-steps", type=int, default=252, help="Time steps per path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    add_heston_args(parser)



def run(args):
    wf = PricingWorkflow("barrier-price", args)
    wf.run_barrier(args)
