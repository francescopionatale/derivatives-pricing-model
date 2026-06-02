from dataclasses import dataclass


@dataclass
class PricingResult:
    price: float
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None


@dataclass
class HedgingResult:
    mean_pnl: float
    std_pnl: float
    var_95: float
    es_95: float
