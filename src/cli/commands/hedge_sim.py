from quant_derivatives.cli.parser import add_standard_market_args, add_heston_args
from quant_derivatives.workflows.hedging_workflow import HedgingWorkflow

def setup_parser(subparsers):
    parser = subparsers.add_parser("hedge-sim", help="Discrete delta hedging simulation")
    add_standard_market_args(parser, require_sigma=False)
    parser.add_argument("--M", type=int, default=1000, help="Number of paths")
    parser.add_argument("--n-steps", type=int, default=252, help="Rebalancing frequency")
    parser.add_argument("--cost", type=float, default=0.0, help="Proportional transaction cost")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    add_heston_args(parser)

def run(args):
    wf = HedgingWorkflow("hedge-sim", args)
    wf.run(args)
