from quant_derivatives.cli.parser import add_standard_market_args
from quant_derivatives.workflows.pricing_workflow import PricingWorkflow

def setup_parser(subparsers):
    parser = subparsers.add_parser("binomial-price", help="Binomial tree pricing")
    add_standard_market_args(parser)
    parser.add_argument("--n-steps", type=int, default=100, help="Number of tree steps")

def run(args):
    wf = PricingWorkflow("binomial-price", args)
    wf.run_binomial(args)
