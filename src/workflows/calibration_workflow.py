import numpy as np
from quant_derivatives.workflows.base import BaseWorkflow
from quant_derivatives.io.loaders import load_quotes_csv
from quant_derivatives.engines.pricing.implied_vol import implied_volatility
from quant_derivatives.engines.calibration.surface import (
    check_no_arbitrage,
    check_put_call_parity,
    calibrate_surface_with_smoothing,
)
from quant_derivatives.engines.calibration.heston import calibrate_heston_to_quotes
from quant_derivatives.utils.heston_params import save_heston_params_json
from quant_derivatives.visualization.plots import plot_implied_vol_surface, plot_implied_vol_smile


class CalibrationWorkflow(BaseWorkflow):
    def run(self, args):
        self.logger.info(f"Loading quotes from {args.input_csv}")
        quotes = load_quotes_csv(args.input_csv)

        results = []
        strikes = []
        maturities = []
        prices = []
        implied_vols = []
        bid_ask_spreads = []

        for q in quotes:
            iv = implied_volatility(q.mid_price, args.S0, q.strike, q.maturity, args.r, q.is_call)
            results.append({
                "strike": q.strike,
                "maturity": q.maturity,
                "mid_price": q.mid_price,
                "is_call": q.is_call,
                "implied_vol": float(iv),
            })
            strikes.append(q.strike)
            maturities.append(q.maturity)
            prices.append(q.mid_price)
            implied_vols.append(float(iv))
            bid_ask_spreads.append(float(q.ask - q.bid) if q.bid is not None and q.ask is not None else np.nan)

        self.logger.info("Checking for static arbitrage...")
        arb_issues = []
        for is_call in (True, False):
            typed_quotes = [q for q in quotes if q.is_call == is_call]
            if not typed_quotes:
                continue
            t_strikes = np.array([q.strike for q in typed_quotes])
            t_maturities = np.array([q.maturity for q in typed_quotes])
            t_prices = np.array([q.mid_price for q in typed_quotes])
            sort_idx = np.lexsort((t_strikes, t_maturities))
            typed_issues = check_no_arbitrage(
                t_strikes[sort_idx],
                t_maturities[sort_idx],
                t_prices[sort_idx],
                is_call=is_call,
                S0=args.S0,
                r=args.r,
            )
            arb_issues.extend(typed_issues)

        arb_issues.extend(check_put_call_parity(quotes, args.S0, args.r))

        output = {
            "calibrated_points": results,
            "arbitrage_issues": arb_issues,
        }

        if arb_issues:
            self.logger.warning(f"Found {len(arb_issues)} arbitrage issues.")
            for issue in arb_issues:
                self.logger.warning(issue)
        else:
            self.logger.info("No static arbitrage issues found.")

        if len(strikes) > 0:
            iv_array = np.array(implied_vols)
            valid = np.isfinite(iv_array)
            if np.count_nonzero(valid) >= 3:
                spreads = np.array(bid_ask_spreads, dtype=float)[valid]
                spreads = spreads if np.any(np.isfinite(spreads)) else None
                grid_K, grid_T, grid_IV = calibrate_surface_with_smoothing(
                    np.array(strikes, dtype=float)[valid],
                    np.array(maturities, dtype=float)[valid],
                    iv_array[valid],
                    bid_ask_spreads=spreads,
                )
                plot_implied_vol_surface(grid_K.ravel(), grid_T.ravel(), grid_IV.ravel())
            else:
                plot_implied_vol_surface(np.array(strikes), np.array(maturities), iv_array)

            plot_implied_vol_smile(np.array(strikes), np.array(maturities), iv_array)

        return output

    def run_heston_calibration(self, args):
        self.logger.info(f"Loading quotes from {args.input_csv}")
        quotes = load_quotes_csv(args.input_csv)
        result = calibrate_heston_to_quotes(
            quotes=quotes,
            S0=args.S0,
            r=args.r,
            n_steps=args.n_steps,
            n_paths=args.M,
            seed=args.seed,
            antithetic=args.antithetic,
            initial_guess={
                "kappa": args.init_kappa,
                "theta": args.init_theta,
                "sigma_v": args.init_sigma_v,
                "rho": args.init_rho,
                "v0": args.init_v0,
            },
            weight_mode=args.weight_mode,
            maxiter=args.maxiter,
        ).to_dict()

        self.logger.info(
            "Calibrated Heston params: "
            f"kappa={result['params']['kappa']:.4f}, theta={result['params']['theta']:.4f}, "
            f"sigma_v={result['params']['sigma_v']:.4f}, rho={result['params']['rho']:.4f}, v0={result['params']['v0']:.4f}"
        )
        self.logger.info(
            f"Calibration RMSE(price)={result['rmse_price']:.6f}, RMSE(iv)={result['rmse_iv']}"
        )
        if args.output_json:
            save_heston_params_json(args.output_json, result)
            self.logger.info(f"Saved calibrated Heston parameters to {args.output_json}")
        return result
