from quant_derivatives.workflows.historical_vol_workflow import HistoricalVolWorkflow

def setup_parser(subparsers):
    parser = subparsers.add_parser("hist-vol", help="Calculate historical realized volatility")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to CSV with historical prices")
    parser.add_argument("--date-col", type=str, default="date", help="Name of the date column")
    parser.add_argument("--price-col", type=str, default="price", help="Name of the price column")
    parser.add_argument("--window", type=int, default=21, help="Rolling window size in days (default: 21)")

def run(args):
    wf = HistoricalVolWorkflow("hist-vol", args)
    wf.run(args)
