from quant_derivatives.cli.parser import add_standard_market_args, add_heston_args
from quant_derivatives.workflows.pricing_workflow import PricingWorkflow


def setup_parser(subparsers):
    parser = subparsers.add_parser("heston-price", help="Heston Monte Carlo pricing")
    add_standard_market_args(parser, require_sigma=False)
    parser.add_argument("--M", type=int, default=10000, help="Number of paths")
    parser.add_argument("--n-steps", type=int, default=252, help="Time steps per path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    add_heston_args(parser, include_model=False)


def run(args):
    wf = PricingWorkflow("heston-price", args)
    wf.run_heston(args)
