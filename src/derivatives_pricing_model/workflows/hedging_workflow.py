import numpy as np
from pathlib import Path
from derivatives_pricing_model.workflows.base import BaseWorkflow
from derivatives_pricing_model.engines.simulation.gbm import simulate_gbm_paths, simulate_gbm_paths_student_t
from derivatives_pricing_model.engines.simulation.heston import simulate_heston_paths
from derivatives_pricing_model.engines.hedging.discrete_hedging import simulate_discrete_hedging
from derivatives_pricing_model.engines.stress.scenario import calculate_var_es
from derivatives_pricing_model.visualization.plots import (
    plot_pnl_distribution,
    plot_pnl_surface,
    plot_pnl_3d_surface,
    plot_pnl_comparison,
    plot_pnl_vs_final_spot,
    plot_pnl_attribution,
    plot_hedging_paths,
)
from derivatives_pricing_model.utils.heston_params import load_heston_params_json


class HedgingWorkflow(BaseWorkflow):
    def _base_heston_params(self, args) -> dict:
        if getattr(args, "params_json", None):
            return load_heston_params_json(args.params_json)
        if args.sigma is None and (args.theta is None or args.v0 is None):
            raise ValueError(
                "Provide --sigma, or provide both --theta and --v0, or use --params-json when using the Heston model."
            )
        ref_var = args.sigma ** 2 if args.sigma is not None else None
        return {
            "kappa": args.kappa,
            "theta": args.theta if args.theta is not None else ref_var,
            "sigma_v": args.sigma_v,
            "rho": args.rho,
            "v0": args.v0 if args.v0 is not None else ref_var,
        }

    def _hedge_sigma(self, args, params: dict | None = None) -> float:
        if args.sigma is not None:
            return float(args.sigma)
        base_params = params if params is not None else self._base_heston_params(args)
        return float(np.sqrt(max(base_params.get("v0", base_params.get("theta", 1e-12)), 1e-12)))

    def _heston_params(self, args, stressed: bool = False) -> dict:
        params = self._base_heston_params(args).copy()
        if stressed:
            params["theta"] *= 1.25
            params["sigma_v"] *= 1.20
            params["v0"] *= 1.50
            params["rho"] = float(np.clip(params["rho"] - 0.10, -0.99, 0.99))
        return params

    def _generate_paths(self, args, stressed: bool = False):
        if args.model == "heston":
            params = self._heston_params(args, stressed=stressed)
            paths, variances = simulate_heston_paths(
                S0=args.S0,
                v0=params["v0"],
                kappa=params["kappa"],
                theta=params["theta"],
                sigma_v=params["sigma_v"],
                rho=params["rho"],
                r=args.r,
                T=args.T,
                n_steps=args.n_steps,
                n_paths=args.M,
                seed=args.seed,
                antithetic=args.antithetic,
            )
            implied_vol_path = np.sqrt(np.clip(variances, 1e-10, None))
            return paths, implied_vol_path, self._hedge_sigma(args, params)

        if args.sigma is None:
            raise ValueError("--sigma is required when using the GBM model.")
        if stressed:
            return (
                simulate_gbm_paths_student_t(args.S0, args.r, args.sigma, args.T, args.n_steps, args.M, df=3.0, seed=args.seed),
                None,
                float(args.sigma),
            )
        return (
            simulate_gbm_paths(args.S0, args.r, args.sigma, args.T, args.n_steps, args.M, args.seed, args.antithetic),
            None,
            float(args.sigma),
        )

    @staticmethod
    def _calc_metrics(pnls, attribution):
        tail = calculate_var_es(pnls, confidence_level=0.95)
        attr_keys = [
            "theta_pnl",
            "gamma_pnl",
            "vega_pnl",
            "vanna_pnl",
            "volga_pnl",
            "cost_pnl",
            "residual_pnl",
        ]
        return {
            "mean_pnl": float(np.mean(pnls)),
            "std_pnl": float(np.std(pnls)),
            "var_95": tail["var"],
            "es_95": tail["es"],
            "ci_95": [float(np.percentile(pnls, 2.5)), float(np.percentile(pnls, 97.5))],
            "avg_realized_vol": float(np.mean(attribution["realized_vol"])),
            "avg_proxy_implied_vol": float(np.mean(attribution["avg_proxy_implied_vol"])),
            "attribution_mean": {k: float(np.mean(attribution[k])) for k in attr_keys},
        }

    def run(self, args):
        from derivatives_pricing_model.visualization import theme

        if getattr(args, "save_plots", None):
            Path(args.save_plots).mkdir(parents=True, exist_ok=True)
            theme.SAVE_DIR = args.save_plots
        plots_enabled = not getattr(args, "no_plots", False)

        self.logger.info(f"Simulating baseline hedging paths under model={args.model}...")
        paths_base, vol_path_base, hedge_sigma_base = self._generate_paths(args, stressed=False)

        self.logger.info("Simulating stress hedging scenario...")
        paths_stress, vol_path_stress, hedge_sigma_stress = self._generate_paths(args, stressed=True)

        self.logger.info("Running discrete delta hedging simulation (Baseline)...")
        res_base = simulate_discrete_hedging(
            paths_base, args.K, args.T, args.r, hedge_sigma_base, args.is_call, args.cost,
            implied_vol_path=vol_path_base, track_paths=plots_enabled,
        )
        pnls_base = res_base["total_pnl"]

        self.logger.info("Running discrete delta hedging simulation (Stress)...")
        res_stress = simulate_discrete_hedging(
            paths_stress, args.K, args.T, args.r, hedge_sigma_stress, args.is_call, args.cost,
            implied_vol_path=vol_path_stress,
        )
        pnls_stress = res_stress["total_pnl"]

        metrics_base = self._calc_metrics(pnls_base, res_base)
        metrics_stress = self._calc_metrics(pnls_stress, res_stress)

        res = {"baseline": metrics_base, "stress": metrics_stress, "model": args.model}

        self.logger.info(
            f"Baseline Mean P&L: {metrics_base['mean_pnl']:.4f}, Std: {metrics_base['std_pnl']:.4f}, "
            f"VaR 95%: {metrics_base['var_95']:.4f}, ES 95%: {metrics_base['es_95']:.4f}"
        )
        self.logger.info(
            f"Stress Mean P&L: {metrics_stress['mean_pnl']:.4f}, Std: {metrics_stress['std_pnl']:.4f}, "
            f"VaR 95%: {metrics_stress['var_95']:.4f}, ES 95%: {metrics_stress['es_95']:.4f}"
        )

        if plots_enabled:
            final_spots_base = paths_base[-1, :]
            plot_pnl_distribution(pnls_base)
            plot_pnl_distribution(pnls_stress)
            plot_pnl_vs_final_spot(final_spots_base, pnls_base, args.K)
            plot_pnl_surface(final_spots_base, pnls_base, strike=args.K)
            plot_pnl_3d_surface(final_spots_base, pnls_base)
            plot_pnl_attribution(res_base)
            plot_pnl_comparison(pnls_base, pnls_stress)
            if "cumulative_pnl_paths" in res_base:
                plot_hedging_paths(res_base["cumulative_pnl_paths"])

        return res
