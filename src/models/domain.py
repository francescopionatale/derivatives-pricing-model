from dataclasses import dataclass


@dataclass
class OptionContract:
    strike: float
    maturity: float
    is_call: bool

@dataclass
class MarketState:
    spot: float
    rate: float
    volatility: float

@dataclass
class OptionQuote:
    strike: float
    maturity: float
    mid_price: float
    bid: float | None = None
    ask: float | None = None
    is_call: bool = True
