from quant_derivatives.cli.parser import add_standard_market_args
from quant_derivatives.workflows.pricing_workflow import PricingWorkflow

def setup_parser(subparsers):
    parser = subparsers.add_parser("mc-price", help="Monte Carlo pricing")
    add_standard_market_args(parser)
    parser.add_argument("--M", type=int, default=10000, help="Number of paths")
    parser.add_argument("--n-steps", type=int, default=252, help="Time steps per path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--antithetic", action="store_true", help="Use antithetic variates")

def run(args):
    wf = PricingWorkflow("mc-price", args)
    wf.run_mc(args)
