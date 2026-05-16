from derivatives_pricing_model.cli.parser import add_standard_market_args, add_plot_args
from derivatives_pricing_model.workflows.pricing_workflow import PricingWorkflow

def setup_parser(subparsers):
    parser = subparsers.add_parser("binomial-price", help="Binomial tree pricing")
    add_standard_market_args(parser)
    parser.add_argument("--n-steps", type=int, default=100, help="Number of tree steps")
    add_plot_args(parser)

def run(args):
    wf = PricingWorkflow("binomial-price", args)
    wf.run_binomial(args)
