import json
from pathlib import Path

from quant_derivatives.workflows.base import BaseWorkflow
from quant_derivatives.engines.risk.optimization import optimize_portfolio


class RiskWorkflow(BaseWorkflow):
    def run(self, args):
        input_path = Path(args.input_json)
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        result = optimize_portfolio(
            current_greeks=payload["current_greeks"],
            available_instruments=payload["available_instruments"],
            target_constraints=payload.get("target_constraints", {}),
            factor_penalties=payload.get("factor_penalties"),
            factor_covariance=payload.get("factor_covariance"),
            risk_aversion=payload.get("risk_aversion", 1.0),
            transaction_costs=payload.get("transaction_costs"),
            bounds=payload.get("bounds"),
            gross_limit=payload.get("gross_limit"),
        )
        self.logger.info("Risk optimization completed")
        self.logger.info(json.dumps(result, indent=2))
        return result
