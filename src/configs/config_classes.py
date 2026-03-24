from dataclasses import dataclass
from typing import Optional

@dataclass
class RunConfig:
    command: str
    seed: Optional[int] = None

@dataclass
class MarketConfig:
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    is_call: bool
