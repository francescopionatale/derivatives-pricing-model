from quant_derivatives.workflows.calibration_workflow import CalibrationWorkflow

def setup_parser(subparsers):
    parser = subparsers.add_parser("calibrate-surface", help="Calibrate implied vol surface")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to quotes CSV")
    parser.add_argument("--S0", type=float, required=True, help="Spot price")
    parser.add_argument("--r", type=float, required=True, help="Risk-free rate")

def run(args):
    wf = CalibrationWorkflow("calibrate-surface", args)
    wf.run(args)
