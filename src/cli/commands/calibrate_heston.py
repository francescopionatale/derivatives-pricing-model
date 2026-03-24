from quant_derivatives.cli.parser import add_heston_calibration_args
from quant_derivatives.workflows.calibration_workflow import CalibrationWorkflow


def setup_parser(subparsers):
    parser = subparsers.add_parser("calibrate-heston", help="Calibrate Heston parameters to vanilla market quotes")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to quotes CSV")
    parser.add_argument("--S0", type=float, required=True, help="Spot price")
    parser.add_argument("--r", type=float, required=True, help="Risk-free rate")
    add_heston_calibration_args(parser)


def run(args):
    wf = CalibrationWorkflow("calibrate-heston", args)
    wf.run_heston_calibration(args)
