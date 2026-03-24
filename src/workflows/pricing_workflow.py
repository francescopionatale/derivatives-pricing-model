import numpy as np
from quant_derivatives.workflows.base import BaseWorkflow
from quant_derivatives.engines.simulation.gbm import simulate_gbm_paths
from quant_derivatives.engines.pricing.black_scholes import bs_price_and_greeks
from quant_derivatives.engines.pricing.implied_vol import implied_volatility
from quant_derivatives.engines.pricing.binomial import binomial_price_and_greeks
from quant_derivatives.engines.pricing.heston_vanilla import heston_vanilla_price_mc
from quant_derivatives.engines.pricing.exotics import (
    price_barrier_mc,
    price_barrier_heston_mc,
    price_lookback_mc,
    price_lookback_heston_mc,
)
from quant_derivatives.visualization.plots import plot_greeks
from quant_derivatives.utils.heston_params import load_heston_params_json


class PricingWorkflow(BaseWorkflow):
    def _resolve_heston_params(self, args):
        if getattr(args, "params_json", None):
            return load_heston_params_json(args.params_json)

        if args.theta is None or args.v0 is None:
            if args.sigma is None:
                raise ValueError(
                    "Provide --sigma, or provide both --theta and --v0, or use --params-json when pricing under Heston."
                )
        return {
            "kappa": args.kappa,
            "theta": args.theta if args.theta is not None else args.sigma ** 2,
            "sigma_v": args.sigma_v,
            "rho": args.rho,
            "v0": args.v0 if args.v0 is not None else args.sigma ** 2,
        }

    def _require_sigma_for_gbm(self, args) -> float:
        if args.sigma is None:
            raise ValueError("--sigma is required when using the GBM model.")
        return float(args.sigma)

    def run_mc(self, args):
        sigma = self._require_sigma_for_gbm(args)
        paths = simulate_gbm_paths(args.S0, args.r, sigma, args.T, args.n_steps, args.M, args.seed, args.antithetic)
        payoffs = np.maximum(paths[-1] - args.K, 0) if args.is_call else np.maximum(args.K - paths[-1], 0)
        discounted_payoffs = np.exp(-args.r * args.T) * payoffs

        price = np.mean(discounted_payoffs)
        std_err = np.std(discounted_payoffs) / np.sqrt(args.M)

        res = {
            "price": float(price),
            "std_err": float(std_err),
            "ci_95": [float(price - 1.96 * std_err), float(price + 1.96 * std_err)],
            "settings": {
                "n_steps": args.n_steps,
                "n_paths": args.M,
                "seed": args.seed,
                "antithetic": args.antithetic,
            },
        }

        self.logger.info(f"MC Price: {price:.4f} +/- {1.96*std_err:.4f}")
        self.logger.info(f"MC Settings: steps={args.n_steps}, paths={args.M}, seed={args.seed}, antithetic={args.antithetic}")
        return res

    def run_bs(self, args):
        sigma = self._require_sigma_for_gbm(args)
        res = bs_price_and_greeks(args.S0, args.K, args.T, args.r, sigma, args.is_call)

        if hasattr(args, 'target_price') and args.target_price is not None:
            iv = implied_volatility(args.target_price, args.S0, args.K, args.T, args.r, args.is_call)
            res['implied_volatility'] = float(iv)
            res['target_price'] = args.target_price
            self.logger.info(f"Implied Volatility for target price {args.target_price}: {iv:.4f}")

        self.logger.info(f"BS Price: {res['price']:.4f}")
        self.logger.info(
            f"BS Greeks: Delta={res['delta']:.4f}, Gamma={res['gamma']:.4f}, Vega={res['vega']:.4f}, Theta={res['theta']:.4f}, Rho={res['rho']:.4f}"
        )
        plot_greeks(res)
        return res

    def run_heston(self, args):
        params = self._resolve_heston_params(args)
        res = heston_vanilla_price_mc(
            S0=args.S0,
            K=args.K,
            T=args.T,
            r=args.r,
            is_call=args.is_call,
            n_steps=args.n_steps,
            n_paths=args.M,
            kappa=params["kappa"],
            theta=params["theta"],
            sigma_v=params["sigma_v"],
            rho=params["rho"],
            v0=params["v0"],
            seed=args.seed,
            antithetic=args.antithetic,
        )
        self.logger.info(f"Heston MC Price: {res['price']:.4f} +/- {1.96*res['std_err']:.4f}")
        self.logger.info(f"Feller condition satisfied: {res['feller_condition']}")
        return res

    def run_barrier(self, args):
        if args.model == "heston":
            params = self._resolve_heston_params(args)
            res = price_barrier_heston_mc(
                S0=args.S0,
                K=args.K,
                T=args.T,
                r=args.r,
                barrier=args.barrier,
                is_up=args.direction == "up",
                is_out=args.barrier_style == "out",
                is_call=args.is_call,
                n_steps=args.n_steps,
                n_paths=args.M,
                kappa=params["kappa"],
                theta=params["theta"],
                sigma_v=params["sigma_v"],
                rho=params["rho"],
                v0=params["v0"],
                seed=args.seed,
                antithetic=args.antithetic,
            )
        else:
            sigma = self._require_sigma_for_gbm(args)
            res = price_barrier_mc(
                S0=args.S0,
                K=args.K,
                T=args.T,
                r=args.r,
                sigma=sigma,
                barrier=args.barrier,
                is_up=args.direction == "up",
                is_out=args.barrier_style == "out",
                is_call=args.is_call,
                n_steps=args.n_steps,
                n_paths=args.M,
                seed=args.seed,
                antithetic=args.antithetic,
            )
        self.logger.info(
            f"Barrier option ({res['model']}, {res['direction']}-{res['style']}) price: {res['price']:.4f} +/- {1.96 * res['std_err']:.4f}"
        )
        self.logger.info(
            f"Barrier={res['barrier']:.4f}, hit ratio={res['barrier_hit_ratio']:.4f}, bridge correction={res.get('brownian_bridge_correction', False)}"
        )
        return res

    def run_lookback(self, args):
        if args.model == "heston":
            params = self._resolve_heston_params(args)
            res = price_lookback_heston_mc(
                S0=args.S0,
                K=args.K,
                T=args.T,
                r=args.r,
                is_floating=args.lookback_style == "floating",
                is_call=args.is_call,
                n_steps=args.n_steps,
                n_paths=args.M,
                kappa=params["kappa"],
                theta=params["theta"],
                sigma_v=params["sigma_v"],
                rho=params["rho"],
                v0=params["v0"],
                seed=args.seed,
                antithetic=args.antithetic,
            )
        else:
            sigma = self._require_sigma_for_gbm(args)
            res = price_lookback_mc(
                S0=args.S0,
                K=args.K,
                T=args.T,
                r=args.r,
                sigma=sigma,
                is_floating=args.lookback_style == "floating",
                is_call=args.is_call,
                n_steps=args.n_steps,
                n_paths=args.M,
                seed=args.seed,
                antithetic=args.antithetic,
            )
        self.logger.info(
            f"Lookback option ({res['model']}, {res['lookback_style']}) price: {res['price']:.4f} +/- {1.96 * res['std_err']:.4f}"
        )
        self.logger.info(
            f"Average path min={res['average_path_min']:.4f}, average path max={res['average_path_max']:.4f}, bridge correction={res.get('brownian_bridge_correction', False)}"
        )
        return res

    def run_binomial(self, args):
        sigma = self._require_sigma_for_gbm(args)
        res = binomial_price_and_greeks(args.S0, args.K, args.T, args.r, sigma, args.n_steps, args.is_call)
        self.logger.info(f"Binomial Price: {res['price']:.4f}")
        self.logger.info(
            f"Binomial Greeks: Delta={res['delta']:.4f}, Gamma={res['gamma']:.4f}, Vega={res['vega']:.4f}, Theta={res['theta']:.4f}, Rho={res['rho']:.4f}"
        )
        plot_greeks(res)
        return res
