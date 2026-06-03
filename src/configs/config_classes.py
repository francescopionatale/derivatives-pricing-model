from dataclasses import dataclass


@dataclass
class RunConfig:
    command: str
    seed: int | None = None

@dataclass
class MarketConfig:
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    is_call: bool
