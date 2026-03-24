from quant_derivatives.workflows.risk_workflow import RiskWorkflow


def setup_parser(subparsers):
    parser = subparsers.add_parser("optimize-risk", help="Portfolio factor-risk optimization from JSON input")
    parser.add_argument("--input-json", required=True, help="Path to JSON file with portfolio, instruments, constraints, and penalties")


def run(args):
    wf = RiskWorkflow("optimize-risk", args)
    wf.run(args)
