from cli.parser import add_standard_market_args, add_plot_args
from workflows.pricing_workflow import PricingWorkflow

def setup_parser(subparsers):
    parser = subparsers.add_parser("bs-price", help="Black-Scholes pricing and Greeks")
    add_standard_market_args(parser)
    parser.add_argument("--target-price", type=float, default=None, help="Target price for implied volatility calculation")
    add_plot_args(parser)

def run(args):
    wf = PricingWorkflow("bs-price", args)
    wf.run_bs(args)
