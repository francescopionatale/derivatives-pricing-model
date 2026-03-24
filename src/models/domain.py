from dataclasses import dataclass
from typing import Optional

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
    bid: Optional[float] = None
    ask: Optional[float] = None
    is_call: bool = True
