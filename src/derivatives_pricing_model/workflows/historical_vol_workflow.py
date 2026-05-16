import numpy as np
from pathlib import Path
from derivatives_pricing_model.workflows.base import BaseWorkflow
from derivatives_pricing_model.io.loaders import load_historical_prices
from derivatives_pricing_model.visualization.plots import plot_historical_volatility

class HistoricalVolWorkflow(BaseWorkflow):
    def run(self, args):
        from derivatives_pricing_model.visualization import theme

        if getattr(args, "save_plots", None):
            Path(args.save_plots).mkdir(parents=True, exist_ok=True)
            theme.SAVE_DIR = args.save_plots
        plots_enabled = not getattr(args, "no_plots", False)

        self.logger.info(f"Loading historical prices from {args.input_csv}")
        df = load_historical_prices(args.input_csv, args.date_col, args.price_col)

        self.logger.info(f"Loaded {len(df)} records. Computing {args.window}-day rolling volatility...")

        df['log_return'] = np.log(df[args.price_col] / df[args.price_col].shift(1))
        df['rolling_vol'] = df['log_return'].rolling(window=args.window).std() * np.sqrt(252)

        valid_data = df.dropna(subset=['rolling_vol'])

        if valid_data.empty:
            self.logger.error("Not enough data to compute rolling volatility.")
            return {}

        latest_vol = valid_data['rolling_vol'].iloc[-1]
        mean_vol = valid_data['rolling_vol'].mean()
        max_vol = valid_data['rolling_vol'].max()
        min_vol = valid_data['rolling_vol'].min()

        res = {
            "latest_volatility": float(latest_vol),
            "mean_volatility": float(mean_vol),
            "max_volatility": float(max_vol),
            "min_volatility": float(min_vol),
            "window": args.window
        }

        self.logger.info(f"Latest {args.window}-day Volatility: {latest_vol:.4f}")

        if plots_enabled:
            plot_historical_volatility(
                valid_data[args.date_col],
                valid_data['rolling_vol'],
                args.window,
                prices=valid_data[args.price_col],
            )

        return res
