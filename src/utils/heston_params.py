from __future__ import annotations

import json
from pathlib import Path


HESTON_PARAM_KEYS = ("kappa", "theta", "sigma_v", "rho", "v0")


def load_heston_params_json(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    params = data.get("params", data)
    missing = [key for key in HESTON_PARAM_KEYS if key not in params]
    if missing:
        raise ValueError(f"Heston params JSON is missing required keys: {missing}")
    return {key: float(params[key]) for key in HESTON_PARAM_KEYS}


def save_heston_params_json(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2))
