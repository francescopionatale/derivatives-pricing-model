from typing import List

import pandas as pd

from quant_derivatives.models.domain import OptionQuote


def load_quotes_csv(filepath: str) -> List[OptionQuote]:
    df = pd.read_csv(filepath)
    required = {"strike", "maturity", "mid_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Quotes CSV is missing required columns: {sorted(missing)}")

    quotes = []
    for _, row in df.iterrows():
        cp_flag = str(row.get('cp_flag', 'C')).upper()
        if cp_flag not in {'C', 'P'}:
            raise ValueError(f"Invalid cp_flag value: {cp_flag}. Expected 'C' or 'P'.")
        quotes.append(OptionQuote(
            strike=row['strike'],
            maturity=row['maturity'],
            mid_price=row['mid_price'],
            bid=row.get('bid'),
            ask=row.get('ask'),
            is_call=cp_flag == 'C'
        ))
    return quotes


def load_historical_prices(filepath: str, date_col: str = 'date', price_col: str = 'price') -> pd.DataFrame:
    """
    Loads historical price data from a CSV file.
    Expects columns for date and price.
    """
    df = pd.read_csv(filepath)
    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(f"CSV must contain '{date_col}' and '{price_col}' columns.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    return df
