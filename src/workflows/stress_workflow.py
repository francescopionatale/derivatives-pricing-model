import numpy as np
from quant_derivatives.workflows.base import BaseWorkflow
from quant_derivatives.engines.simulation.gbm import simulate_gbm_paths
from quant_derivatives.engines.simulation.heston import simulate_heston_paths
from quant_derivatives.engines.stress.scenario import (
    generate_student_t_paths,
    calculate_var_es,
    apply_spot_vol_shock,
    generate_short_convexity_scenario,
)
from quant_derivatives.engines.hedging.discrete_hedging import simulate_discrete_hedging
from quant_derivatives.utils.heston_params import load_heston_params_json


class StressWorkflow(BaseWorkflow):
    def _base_heston_params(self, args) -> dict:
        if getattr(args, "params_json", None):
            return load_heston_params_json(args.params_json)
        if args.sigma is None and (args.theta is None or args.v0 is None):
            raise ValueError(
                "Provide --sigma, or provide both --theta and --v0, or use --params-json when using the Heston model."
            )
        ref_var = args.sigma ** 2 if args.sigma is not None else None
        return dict(
            kappa=args.kappa,
            theta=args.theta if args.theta is not None else ref_var,
            sigma_v=args.sigma_v,
            rho=args.rho,
            v0=args.v0 if args.v0 is not None else ref_var,
        )

    def _hedge_sigma(self, args, params: dict | None = None) -> float:
        if args.sigma is not None:
            return float(args.sigma)
        base_params = params if params is not None else self._base_heston_params(args)
        return float(np.sqrt(max(base_params.get("v0", base_params.get("theta", 1e-12)), 1e-12)))

    def _heston_params(self, args, shocked_sigma: float | None = None, stressed: bool = False):
        params = self._base_heston_params(args).copy()
        if shocked_sigma is not None:
            params["theta"] = max(params["theta"], shocked_sigma ** 2)
            params["v0"] = max(params["v0"], shocked_sigma ** 2)
        if stressed:
            params["theta"] *= 1.35
            params["v0"] *= 1.60
            params["sigma_v"] *= 1.25
            params["rho"] = float(np.clip(params["rho"] - 0.15, -0.99, 0.99))
        return params

    def _simulate(self, args, scenario: str):
        base_sigma = self._hedge_sigma(args)
        if args.model == "heston":
            if scenario == "gaussian":
                params = self._heston_params(args)
                paths, variances = simulate_heston_paths(args.S0, params["v0"], params["kappa"], params["theta"], params["sigma_v"], params["rho"], args.r, args.T, args.n_steps, args.M, args.seed, args.antithetic)
                return paths, np.sqrt(np.clip(variances, 1e-10, None)), base_sigma
            if scenario == "student_t":
                params = self._heston_params(args, stressed=True)
                paths, variances = simulate_heston_paths(args.S0, params["v0"], params["kappa"], params["theta"], params["sigma_v"], params["rho"], args.r, args.T, args.n_steps, args.M, args.seed, args.antithetic)
                return paths, np.sqrt(np.clip(variances, 1e-10, None)), self._hedge_sigma(args, params)
            if scenario == "spot_vol_shock":
                shocked_S0, shocked_sigma = apply_spot_vol_shock(args.S0, base_sigma, args.spot_shock, args.vol_shock)
                params = self._heston_params(args, shocked_sigma=shocked_sigma, stressed=True)
                paths, variances = simulate_heston_paths(shocked_S0, params["v0"], params["kappa"], params["theta"], params["sigma_v"], params["rho"], args.r, args.T, args.n_steps, args.M, args.seed, args.antithetic)
                return paths, np.sqrt(np.clip(variances, 1e-10, None)), shocked_sigma
            if scenario == "short_convexity":
                paths = generate_short_convexity_scenario(args.S0, args.r, base_sigma, args.T, args.n_steps, args.M, args.seed)
                return paths, None, base_sigma

        if args.sigma is None:
            raise ValueError("--sigma is required when using the GBM model.")
        if scenario == "gaussian":
            return simulate_gbm_paths(args.S0, args.r, args.sigma, args.T, args.n_steps, args.M, args.seed, args.antithetic), None, args.sigma
        if scenario == "student_t":
            return generate_student_t_paths(args.S0, args.r, args.sigma, args.T, args.n_steps, args.M, args.df, args.seed), None, args.sigma
        if scenario == "spot_vol_shock":
            shocked_S0, shocked_sigma = apply_spot_vol_shock(args.S0, args.sigma, args.spot_shock, args.vol_shock)
            return simulate_gbm_paths(shocked_S0, args.r, shocked_sigma, args.T, args.n_steps, args.M, args.seed, args.antithetic), None, shocked_sigma
        if scenario == "short_convexity":
            return generate_short_convexity_scenario(args.S0, args.r, args.sigma, args.T, args.n_steps, args.M, args.seed), None, args.sigma
        raise ValueError(f"Unknown scenario: {scenario}")

    @staticmethod
    def _metrics(result):
        pnls = result["total_pnl"]
        tail = calculate_var_es(pnls, confidence_level=0.95)
        return {
            "mean_pnl": float(np.mean(pnls)),
            "std_pnl": float(np.std(pnls)),
            "var_95": tail["var"],
            "es_95": tail["es"],
            "avg_realized_vol": float(np.mean(result["realized_vol"])),
            "avg_proxy_implied_vol": float(np.mean(result["avg_proxy_implied_vol"])),
        }

    def run(self, args):
        scenarios = ["gaussian", "student_t", "spot_vol_shock", "short_convexity"]
        results = {}
        for scenario in scenarios:
            self.logger.info(f"Running scenario={scenario} under model={args.model}...")
            paths, vol_path, hedge_sigma = self._simulate(args, scenario)
            hedge_res = simulate_discrete_hedging(paths, args.K, args.T, args.r, hedge_sigma, args.is_call, args.cost, implied_vol_path=vol_path)
            results[scenario] = self._metrics(hedge_res)

        self.logger.info(f"Gaussian VaR 95%: {results['gaussian']['var_95']:.4f}")
        self.logger.info(f"Gaussian ES 95%: {results['gaussian']['es_95']:.4f}")
        self.logger.info(f"Student-t / stressed VaR 95%: {results['student_t']['var_95']:.4f}")
        self.logger.info(f"Student-t / stressed ES 95%: {results['student_t']['es_95']:.4f}")
        self.logger.info(f"Finite spot-vol shock VaR 95%: {results['spot_vol_shock']['var_95']:.4f}")
        self.logger.info(f"Short-convexity ES 95%: {results['short_convexity']['es_95']:.4f}")
        return results
