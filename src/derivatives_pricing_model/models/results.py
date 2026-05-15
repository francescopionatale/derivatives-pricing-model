from dataclasses import dataclass
from typing import Optional


@dataclass
class PricingResult:
    price: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None


@dataclass
class HedgingResult:
    mean_pnl: float
    std_pnl: float
    var_95: float
    es_95: float
