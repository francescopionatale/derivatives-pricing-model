from quant_derivatives.cli.parser import add_standard_market_args, add_heston_args
from quant_derivatives.workflows.stress_workflow import StressWorkflow

def setup_parser(subparsers):
    parser = subparsers.add_parser("stress-run", help="Stress testing with heavy tails")
    add_standard_market_args(parser, require_sigma=False)
    parser.add_argument("--M", type=int, default=1000, help="Number of paths")
    parser.add_argument("--n-steps", type=int, default=252, help="Steps")
    parser.add_argument("--cost", type=float, default=0.0, help="Transaction cost")
    parser.add_argument("--df", type=float, default=4.0, help="Degrees of freedom for Student-t")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--spot-shock", type=float, default=-0.10, help="Relative finite spot shock for dedicated stress scenario")
    parser.add_argument("--vol-shock", type=float, default=0.05, help="Absolute volatility shock for dedicated stress scenario")
    add_heston_args(parser)

def run(args):
    wf = StressWorkflow("stress-run", args)
    wf.run(args)
