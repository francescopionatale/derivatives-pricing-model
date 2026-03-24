import numpy as np
from quant_derivatives.workflows.base import BaseWorkflow
from quant_derivatives.io.loaders import load_historical_prices
from quant_derivatives.visualization.plots import plot_historical_volatility

class HistoricalVolWorkflow(BaseWorkflow):
    def run(self, args):
        self.logger.info(f"Loading historical prices from {args.input_csv}")
        df = load_historical_prices(args.input_csv, args.date_col, args.price_col)
        
        self.logger.info(f"Loaded {len(df)} records. Computing {args.window}-day rolling volatility...")
        
        # Calculate daily log returns
        df['log_return'] = np.log(df[args.price_col] / df[args.price_col].shift(1))
        
        # Calculate rolling standard deviation (annualized)
        # Assuming 252 trading days in a year
        df['rolling_vol'] = df['log_return'].rolling(window=args.window).std() * np.sqrt(252)
        
        # Drop NaN values for plotting and reporting
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
        
        # Generate Plot
        plot_historical_volatility(valid_data[args.date_col], valid_data['rolling_vol'], args.window)
        
        return res
